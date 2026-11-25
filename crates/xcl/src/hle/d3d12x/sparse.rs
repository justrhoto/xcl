//! The title's own memory, seen by the GPU: sparse windows.
//!
//! A console title keeps GPU data in memory it reserves and commits itself (XDefiant: one
//! 8 GB reservation of 64 KiB pages it commits in steps and places buffers in) – on the
//! console all of it is GPU-visible at the same addresses. XCL gives such reservations
//! addresses in the *sparse windows* of the unified-memory block (`gpu::uma`: windows 0..2
//! in `for_runtime`), so translated shaders reach them through `mem`…`mem2` like any unified
//! memory. On the GPU each sparse window is a reserved (tiled) buffer of 4 GiB whose 64 KiB
//! tiles are mapped onto video-memory heaps as the title commits pages.
//! The CPU's view stays the title's memory: what the CPU wrote (write-watch) is copied into
//! the tiles before the title's next ExecuteCommandLists via an asynchronous ring-buffered
//! staging upload queue, and the title's queues GPU-wait on the upload fence (no CPU stall).
//! Only the bytes the CPU changed go (64-byte chunks against a hash of what was sent last,
//! `changed_chunks`): the rest of a written page may hold what the GPU wrote there (counters,
//! indirect arguments) and must not be overwritten with the CPU's stale view of it.
//! Buffers the title places there get reserved resources over the same tiles, so PC-side
//! copies, views and barriers see the same memory.

use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::ffi::c_void;
use std::hash::{BuildHasherDefault, Hasher};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Mutex;

use windows::core::Interface;
use windows::Win32::Foundation::{CloseHandle, HANDLE};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;
use windows_sys::Win32::System::Memory::{
    VirtualAlloc, VirtualFree, VirtualQuery, MEMORY_BASIC_INFORMATION, MEM_COMMIT, MEM_DECOMMIT, PAGE_GUARD,
    PAGE_NOACCESS, PAGE_READWRITE,
};

const TILE: usize = 0x1_0000;
/// tiles of a video-memory pool heap (64 MiB)
const POOL_TILES: u32 = 1024;
/// tiles decommitted while no title work was submitted that stay mapped (`State::parked`) before
/// they are unmapped together: 256 MB of video memory
const PARKED_MAX: usize = 4096;
/// tiles waiting for their mapping call (`State::pending`) before they are sent at once: 256 MB
const PENDING_MAX: usize = 4096;
/// bytes of the ring-buffered upload staging buffer
const STAGING: usize = 64 << 20;
/// alignment of sub-allocations inside the upload staging ring (write-combined cacheline)
const RING_ALIGN: usize = 64;
/// number of in-flight upload command allocators / lists
const UPLOAD_SLOTS: usize = 16;
/// bytes of the dedicated pre-zeroed buffer for tile/hole clears (1 MiB)
const ZERO_BUF: usize = 1 << 20;
/// bytes of the GPU→CPU readback staging buffer
const READBACK_STAGING: usize = 16 << 20;

/// changed runs of fewer bytes go up together in a compute dispatch (`scatter`) when there are
/// this many of them, of this many chunks at most (4 MiB, so the ring fills in pieces)
const SCATTER_RUN: usize = 4 << 10;
const SCATTER_FROM: usize = 16;
const SCATTER_CHUNKS: usize = 1 << 16;

const PAGE: usize = 0x1000;
const PAGES_PER_TILE: usize = TILE / PAGE;
const CHUNK: usize = super::scatter::CHUNK;
const CHUNKS: usize = PAGE / CHUNK;

/// Fast non-cryptographic integer hasher for page/tile lookup maps on the hot submit path.
#[derive(Default)]
struct FxHasher(u64);

impl Hasher for FxHasher {
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = (self.0.rotate_left(5) ^ b as u64).wrapping_mul(0x517C_C1B7_2722_0A95);
        }
    }
    #[inline]
    fn write_usize(&mut self, i: usize) {
        self.0 = (self.0.rotate_left(5) ^ i as u64).wrapping_mul(0x517C_C1B7_2722_0A95);
    }
    #[inline]
    fn write_u32(&mut self, i: u32) {
        self.0 = (self.0.rotate_left(5) ^ i as u64).wrapping_mul(0x517C_C1B7_2722_0A95);
    }
    #[inline]
    fn write_u64(&mut self, i: u64) {
        self.0 = (self.0.rotate_left(5) ^ i).wrapping_mul(0x517C_C1B7_2722_0A95);
    }
    #[inline]
    fn finish(&self) -> u64 {
        // Tile addresses are 64 KiB aligned. Move mixed bits into the low bucket-index bits.
        self.0.rotate_right(23)
    }
}

type FxHashMap<K, V> = HashMap<K, V, BuildHasherDefault<FxHasher>>;

struct TileEntry {
    pool: (u32, u32),
    /// bit `i` set => 4 KiB page `i` of this 64 KiB tile is uncommitted (a hole)
    holes: u16,
    /// per-4 KiB-page hashes of the 64-byte chunks last sent to the GPU (`None` = all zero)
    sent: [Option<Box<[u64; CHUNKS]>>; PAGES_PER_TILE],
}

impl TileEntry {
    fn with_holes(pool: (u32, u32), holes: u16) -> Self {
        Self { pool, holes, sent: std::array::from_fn(|_| None) }
    }
}

/// Query actual OS page commitment across the 64 KiB tile at `tile`.
/// Bit `i` in the returned mask is set whenever 4 KiB page `tile + i * PAGE` is not committed
/// and accessible (`MEM_RESERVE`, `MEM_FREE`, `PAGE_NOACCESS`, `PAGE_GUARD`, etc.).
unsafe fn query_tile_holes(tile: usize) -> u16 {
    let mut holes = 0xFFFFu16;
    let end = tile.saturating_add(TILE);
    let mut cur = tile;
    while cur < end {
        let mut mbi: MEMORY_BASIC_INFORMATION = std::mem::zeroed();
        // SAFETY: `VirtualQuery` inspects process virtual memory metadata without dereferencing `cur`.
        let n = VirtualQuery(cur as *const c_void, &mut mbi, std::mem::size_of::<MEMORY_BASIC_INFORMATION>());
        if n == 0 || mbi.RegionSize == 0 {
            break;
        }
        let region_end = (mbi.BaseAddress as usize).saturating_add(mbi.RegionSize).min(end);
        let accessible = mbi.State == MEM_COMMIT && mbi.Protect != 0 && (mbi.Protect & (PAGE_NOACCESS | PAGE_GUARD)) == 0;
        if accessible {
            let mut pg = cur;
            while pg < region_end {
                holes &= !(1u16 << ((pg - tile) / PAGE));
                pg += PAGE;
            }
        }
        if region_end <= cur {
            break;
        }
        cur = region_end;
    }
    holes
}

/// Update `mapped`, `free`, and `to_zero` for a decommit of `[a, a + n)`.
/// Returns the base addresses of tiles that became completely uncommitted (`holes == 0xFFFF`)
/// and were removed from `mapped` (ready to be unmapped from their sparse window buffers).
fn decommit_tiles(
    mapped: &mut FxHashMap<usize, TileEntry>,
    pinned: &FxHashMap<(u32, u32), u32>,
    free: &mut Vec<(u32, u32)>,
    to_zero: &mut BTreeSet<(usize, usize)>,
    a: usize,
    n: usize,
) -> Vec<usize> {
    if n == 0 {
        return Vec::new();
    }
    let mut unmapped: Vec<usize> = Vec::new();
    let mut pg = a & !(PAGE - 1);
    let end = a.saturating_add(n);
    while pg < end {
        let t = pg & !(TILE - 1);
        let tile_end = t.saturating_add(TILE).min(end);
        if let Some(entry) = mapped.get_mut(&t) {
            let mut newly_holed = 0u16;
            let mut p = pg;
            while p < tile_end {
                let idx = (p - t) / PAGE;
                entry.sent[idx] = None;
                let bit = 1u16 << idx;
                if entry.holes & bit == 0 {
                    entry.holes |= bit;
                    newly_holed |= bit;
                }
                p += PAGE;
            }
            if entry.holes == 0xFFFF {
                unmapped.push(t);
            } else if newly_holed != 0 {
                for idx in 0..PAGES_PER_TILE {
                    if newly_holed & (1u16 << idx) != 0 {
                        to_zero.insert((t + idx * PAGE, PAGE));
                    }
                }
            }
        }
        if tile_end <= pg {
            break;
        }
        pg = tile_end;
    }
    for &t in &unmapped {
        if let Some(entry) = mapped.remove(&t) {
            if !pinned.contains_key(&entry.pool) {
                free.push(entry.pool);
            }
        }
    }
    // (their pieces only: going over all of them for each page decommitted while a title loads,
    // with no submit to clear them, grew with the square of the pages)
    for &t in &unmapped {
        while let Some(&piece) = to_zero.range((t, 0)..(t + TILE, 0)).next() {
            to_zero.remove(&piece);
        }
    }
    unmapped
}

/// The pieces to clear (`State::to_zero`) as runs: neighbours and pieces inside another together.
fn zero_runs(pieces: BTreeSet<(usize, usize)>) -> Vec<(usize, usize)> {
    let mut out: Vec<(usize, usize)> = Vec::new();
    for (a, n) in pieces {
        match out.last_mut() {
            Some(l) if a <= l.0 + l.1 => l.1 = l.1.max(a + n - l.0),
            _ => out.push((a, n)),
        }
    }
    out
}

struct UploadSlot {
    alloc: ID3D12CommandAllocator,
    list: ID3D12GraphicsCommandList,
    fence_val: u64,
}

/// Pool tiles unmapped from their windows, free once `marks` are reached (#2345).
struct RetiredTiles {
    tiles: Vec<(u32, u32)>,
    /// Fence marks on `self.queue` and every known title queue (`None` = signal failed, keep held).
    marks: Option<Vec<(ID3D12Fence, u64)>>,
}

/// A copy back on `State::rbq`: its list, its readback buffer, the `rb_fence` value it waits
/// for, the `rbq` fence value that marks it done, and the copies into the title's memory
/// (address, offset in `buf`, bytes).
struct RbJob {
    alloc: ID3D12CommandAllocator,
    list: ID3D12GraphicsCommandList,
    buf: ID3D12Resource,
    ptr: *mut u8,
    cap: usize,
    rb_v: u64,
    done_v: u64,
    copies: Vec<(usize, usize, usize)>,
    /// the type of queue `list` is for (a copy back runs on the title's own queue)
    ty: D3D12_COMMAND_LIST_TYPE,
    /// the title queue's fences for it (`State::rb_queue_fences`): the first reaches `rb_v` when
    /// the lists it follows are done, the second `done_v` when the copy is. None: `rb_fence` and
    /// the readback queue's fence (the form before)
    fences: Option<(ID3D12Fence, ID3D12Fence)>,
    /// place among the submissions' jobs (`State::job_seq`): copies and immediates land in it
    seq: u64,
}

/// In an immediate's byte count: the cell is a counter that the title's queues write in turn,
/// with values that only rise as they are recorded. A list that runs late – a retail title's
/// compute jobs, tens of frames after their recording when its submits are slow – writes an
/// earlier value over a later one; the title's CPU waits for "at least" the later one, which
/// was reached. The CPU's view keeps the larger value.
pub(super) const IMM_KEEP_MAX: u8 = 0x80;

/// Immediates the title's lists wrote into a window (`list_x::imm_by_cpu`): stored in the
/// title's memory once `fence` reaches `value` (the lists are done on the GPU).
struct ImmJob {
    /// place among the copies back (`RbJob::seq`)
    seq: u64,
    fence: ID3D12Fence,
    value: u64,
    /// (address, value, bytes)
    writes: Vec<(usize, u64, u8)>,
}

/// How far `drain_readbacks` goes
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Drain {
    /// only copies the GPU has finished
    Finished,
    /// also those whose title work is done (waiting for the copy itself): before a CPU read
    /// the title made sure of, e.g. after its wait for the value a list wrote
    TitleDone,
}

struct State {
    dev: ID3D12Device,
    queue: ID3D12CommandQueue,
    fence: ID3D12Fence,
    value: u64,
    /// reusable auto-reset Win32 event for fence waits (no per-wait CreateEventW/CloseHandle)
    event: HANDLE,
    /// fence signalled on title queues after their lists, so a copy back waits for them
    rb_fence: ID3D12Fence,
    rb_value: u64,
    /// title queue -> the two fences it signals around a copy back, and the last value. One set
    /// a queue: with one fence for all, a queue that got to its Signal first let the copies of a
    /// queue still waiting run early
    rb_queue_fences: FxHashMap<usize, (ID3D12Fence, ID3D12Fence, u64)>,
    /// copies back after the title's submissions (`readback_after_execute`): on a queue of their
    /// own (`rbq`, None: on `queue`, waited for on the CPU), landing in the title's memory once
    /// done (`drain_readbacks`). The title's queues can wait for work the title submits after
    /// this one, from the same thread: waiting here for its lists stopped that submission – and,
    /// on `queue`, every upload behind it (`before_execute` waiting for ring room).
    rbq: Option<(ID3D12CommandQueue, ID3D12Fence)>,
    rbq_value: u64,
    /// submitted copies back, oldest first, and finished ones to reuse
    rb_jobs: VecDeque<RbJob>,
    rb_pool: Vec<RbJob>,
    /// immediates of the title's submissions, oldest first, and the count that orders them
    /// with `rb_jobs`
    imm_jobs: VecDeque<ImmJob>,
    job_seq: u64,
    /// (window start, reserved buffer) of each sparse window
    windows: Vec<(usize, ID3D12Resource)>,
    /// video-memory pool heaps (None: given back, its index free for the next one)
    pools: Vec<Option<ID3D12Heap>>,
    free: Vec<(u32, u32)>,
    /// pools all of whose tiles came free: released once the GPU is past `value` and a moment
    /// went by (titles' queues may still read what was mapped there)
    retired: Vec<(ID3D12Heap, u64, std::time::Instant)>,
    /// tile address → pool tile + uncommitted page hole mask + per-page sent chunk hashes
    mapped: FxHashMap<usize, TileEntry>,
    /// pool tiles also used by placed buffers (not freed while they are)
    pinned: FxHashMap<(u32, u32), u32>,
    /// tile address → pool tile of tiles decommitted since the last title submit, still mapped
    /// in their window: unmapped together (`flush_parked`) before the next submit, when the free
    /// list runs out, or past `PARKED_MAX`; a commit there before that takes its tile back with
    /// no new mapping. (A retail title moving physical pages one at a time while it loads
    /// decommits and commits again thousands of tiles a second, and each `UpdateTileMappings`
    /// took ~0.15 ms.)
    parked: FxHashMap<usize, (u32, u32)>,
    /// unmapped parked pool tiles awaiting `self.queue` and in-flight title queues before entering `free`
    quarantined: VecDeque<RetiredTiles>,
    /// per-title-queue fences used to mark in-flight GPU work (`note_queue`)
    queue_fences: FxHashMap<usize, (ID3D12CommandQueue, ID3D12Fence, u64)>,
    /// tile address → pool tile of decommitted tiles a placed buffer uses: they stay mapped in
    /// their window until the buffer is gone. Memory committed there again is then the same for
    /// the window (what shaders write) and the buffer (what an ExecuteIndirect or a copy of the
    /// title reads); with a new tile in the window the buffer kept the old one, with old bytes
    kept: FxHashMap<usize, (u32, u32)>,
    /// tile addresses given a pool tile in `mapped` whose `UpdateTileMappings` has not been sent
    /// yet: mapped together (`flush_pending`, one call per window) before the next title submit,
    /// before XCL's own GPU work on the windows, or past `PENDING_MAX`. (A retail title starting
    /// on a Series S profile maps 4.3 GB of physical pages in small pieces: one mapping call
    /// each, ~0.4 ms, held its loading thread ~25 s and its stall watchdog ended it.)
    pending: Vec<usize>,
    /// placed buffers → the tiles they use (address, pool tile)
    buffers: FxHashMap<usize, Vec<(usize, (u32, u32))>>,
    /// reused tiles / decommitted partial pages to clear: (address, bytes)
    to_zero: BTreeSet<(usize, usize)>,
    /// ring-buffered upload staging heap (`STAGING` bytes, persistently mapped)
    staging: Option<(ID3D12Resource, *mut u8)>,
    /// monotonic ring-buffer write/submitted/retired byte cursors
    ring_write: u64,
    ring_submitted: u64,
    ring_free: u64,
    /// submitted ring chunks awaiting GPU completion: `(fence_val, ring_end_pos)`
    inflight: VecDeque<(u64, u64)>,
    /// dedicated 1 MiB pre-zeroed buffer for tile/hole clears (zero staging bandwidth)
    zero_buf: Option<ID3D12Resource>,
    /// ring of async upload command allocators + lists
    slots: Vec<UploadSlot>,
    next_slot: usize,
    /// dedicated readback staging buffer + command allocator / list
    readback: Option<(ID3D12Resource, *mut u8)>,
    rb_alloc: ID3D12CommandAllocator,
    rb_list: ID3D12GraphicsCommandList,
    rb_last_fence: u64,
    /// the fence value each title queue last waited for
    waited: FxHashMap<usize, u64>,
    /// `mapped` as runs of neighbouring tiles (start -> end), kept with it: sorted out of all of
    /// them again (a retail title has 150,000) for each look behind a commit
    runs: BTreeMap<usize, usize>,
    /// reusable buffer for dirty runs in `before_execute` (avoids per-submit heap alloc)
    scratch_runs: Vec<(usize, usize)>,
}

impl Drop for State {
    fn drop(&mut self) {
        if !self.event.is_invalid() {
            unsafe {
                let _ = CloseHandle(self.event);
            }
        }
    }
}

const K0: u64 = 0x9E37_79B9_7F4A_7C15;
const K1: u64 = 0xBF58_476D_1CE4_E5B9;
const K2: u64 = 0x94D0_49BB_1331_11EB;
const K3: u64 = 0x243F_6A88_85A3_08D3;

/// 4-way parallel hash over 8 `u64` words (64 bytes): breaks serial multiply latency chain.
#[inline(always)]
const fn hash_words(w: [u64; 8]) -> u64 {
    let a = (w[0] ^ K3).wrapping_mul(K0) ^ w[4].wrapping_mul(K1);
    let b = (w[1] ^ K0).wrapping_mul(K1) ^ w[5].wrapping_mul(K2);
    let c = (w[2] ^ K1).wrapping_mul(K2) ^ w[6].wrapping_mul(K3);
    let d = (w[3] ^ K2).wrapping_mul(K3) ^ w[7].wrapping_mul(K0);
    let h = a.rotate_left(27).wrapping_add(b.rotate_left(13)) ^ c.rotate_left(41).wrapping_add(d.rotate_left(53));
    h.wrapping_mul(K0) ^ (h >> 32)
}

const ZERO_HASH: u64 = hash_words([0; 8]);
const ZERO_HASHES: [u64; CHUNKS] = [ZERO_HASH; CHUNKS];

#[inline(always)]
fn read_chunk_words(c: &[u8]) -> [u64; 8] {
    debug_assert_eq!(c.len(), CHUNK);
    let p = c.as_ptr() as *const [u8; 8];
    unsafe {
        [
            u64::from_le_bytes(p.add(0).read_unaligned()),
            u64::from_le_bytes(p.add(1).read_unaligned()),
            u64::from_le_bytes(p.add(2).read_unaligned()),
            u64::from_le_bytes(p.add(3).read_unaligned()),
            u64::from_le_bytes(p.add(4).read_unaligned()),
            u64::from_le_bytes(p.add(5).read_unaligned()),
            u64::from_le_bytes(p.add(6).read_unaligned()),
            u64::from_le_bytes(p.add(7).read_unaligned()),
        ]
    }
}

/// hash of 64 bytes
#[inline(always)]
fn chunk_hash(c: &[u8]) -> u64 {
    hash_words(read_chunk_words(c))
}

#[inline(always)]
fn push_run(out: &mut Vec<(usize, usize)>, addr: usize, len: usize) {
    match out.last_mut() {
        Some(last) if last.0 + last.1 == addr => last.1 += len,
        _ => out.push((addr, len)),
    }
}

/// Copy `len` bytes from `src` to offset `at` of the title's page `pg`, 64-byte chunk by chunk:
/// a chunk the CPU wrote since it was last sent keeps its (newer) bytes and goes up with the next
/// look; a chunk as it was sent gets what the GPU copied back, and its hash with it.
unsafe fn land_page(slot: &mut Option<Box<[u64; CHUNKS]>>, pg: usize, at: usize, src: *const u8, len: usize) {
    let sent = slot.get_or_insert_with(|| Box::new(ZERO_HASHES));
    let end = at + len;
    let mut o = at;
    while o < end {
        let k = o / CHUNK;
        let chunk_end = ((k + 1) * CHUNK).min(end);
        let chunk = std::slice::from_raw_parts((pg + k * CHUNK) as *const u8, CHUNK);
        if chunk_hash(chunk) == sent[k] {
            std::ptr::copy_nonoverlapping(src.add(o - at), (pg + o) as *mut u8, chunk_end - o);
            sent[k] = chunk_hash(std::slice::from_raw_parts((pg + k * CHUNK) as *const u8, CHUNK));
        }
        o = chunk_end;
    }
}

/// Scan a 4 KiB page at `pg`, appending any changed 64-byte chunk runs directly to `out`
/// (zero per-page heap allocation; skips allocating `slot` completely when a fresh zero page
/// is still all zeros).
#[inline]
fn scan_page(page: &[u8], pg: usize, slot: &mut Option<Box<[u64; CHUNKS]>>, out: &mut Vec<(usize, usize)>) {
    if let Some(sent) = slot.as_deref_mut() {
        for (k, c) in page.chunks_exact(CHUNK).enumerate() {
            let h = chunk_hash(c);
            if h == sent[k] {
                continue;
            }
            sent[k] = h;
            push_run(out, pg + k * CHUNK, CHUNK);
        }
        return;
    }
    // Page has never had non-zero data sent: skip zero chunks with a cheap OR-reduction and
    // allocate the hash array only if at least one chunk is non-zero.
    let mut chunks = page.chunks_exact(CHUNK).enumerate();
    while let Some((k, c)) = chunks.next() {
        let w = read_chunk_words(c);
        if (w[0] | w[1] | w[2] | w[3] | w[4] | w[5] | w[6] | w[7]) == 0 {
            continue;
        }
        let h = hash_words(w);
        if h == ZERO_HASH {
            continue;
        }
        let sent = slot.insert(Box::new(ZERO_HASHES));
        sent[k] = h;
        push_run(out, pg + k * CHUNK, CHUNK);
        for (k2, c2) in chunks {
            let h2 = chunk_hash(c2);
            if h2 == ZERO_HASH {
                continue;
            }
            sent[k2] = h2;
            push_run(out, pg + k2 * CHUNK, CHUNK);
        }
        break;
    }
}

/// The byte ranges (offset, length) of `page` whose 64-byte chunks differ from `sent`
/// (updated to the page's chunks).
pub fn changed_chunks(page: &[u8], sent: &mut [u64; CHUNKS]) -> Vec<(usize, usize)> {
    let mut out: Vec<(usize, usize)> = Vec::new();
    for (k, c) in page.chunks_exact(CHUNK).enumerate() {
        let h = chunk_hash(c);
        if h == sent[k] {
            continue;
        }
        sent[k] = h;
        push_run(&mut out, k * CHUNK, CHUNK);
    }
    out
}

/// Plan a contiguous sub-allocation `(aligned_pos, chunk_len)` inside a ring buffer of `cap`
/// bytes starting at monotonic cursor `ring_write`, keeping each chunk `<= cap / 2` and
/// avoiding mid-copy wrap-around without wasting large tail gaps.
#[inline]
fn plan_ring_chunk(ring_write: u64, want: usize, cap: usize) -> (u64, usize) {
    let mut chunk = want.min(cap / 2);
    let mut aligned = (ring_write + (RING_ALIGN as u64 - 1)) & !(RING_ALIGN as u64 - 1);
    let off = (aligned as usize) % cap;
    if off + chunk > cap {
        let tail = cap - off;
        if tail >= PAGE {
            chunk = tail;
        } else {
            aligned += tail as u64;
        }
    }
    (aligned, chunk)
}

// D3D12 objects are free-threaded; staging pointers and event handle are only used under STATE lock
unsafe impl Send for State {}

static STATE: Mutex<Option<State>> = Mutex::new(None);

/// The state's queue again, outside `STATE` (`queue`). `STATE` is held around D3D12 calls (an
/// upload's allocator `Reset` in `begin_slot`), and `late_release` asks for the queue from a
/// resource's last `Release`, which D3D12 can make inside a call of its own on another thread
/// (a list's `Reset`, with `XCL_DRED=1` on every start): the two then waited for each other.
static QUEUE: Mutex<Option<ID3D12CommandQueue>> = Mutex::new(None);

/// (bytes of video memory in pool heaps, of those mapped) – `XCL_ACTIVITY`
pub fn pool_bytes() -> (usize, usize) {
    let g = lock_state();
    let Some(s) = g.as_ref() else { return (0, 0) };
    let pools = s.pools.iter().filter(|p| p.is_some()).count() * POOL_TILES as usize * TILE;
    (pools, pools.saturating_sub(s.free.len() * TILE))
}
/// commits before the device existed: the pages committed, as disjoint page ranges (start →
/// end) merged with their neighbours, so a commit or decommit costs O(log n) and not a scan of
/// every page committed so far. Like `committed` / `decommitted` with a state, a range counts
/// every page it touches; `init` commits them once the windows exist.
static EARLY: Mutex<BTreeMap<usize, usize>> = Mutex::new(BTreeMap::new());

/// The pages `[a, a + n)` touches, as `[start, end)`.
fn page_span(a: usize, n: usize) -> (usize, usize) {
    (a & !(PAGE - 1), a.saturating_add(n).saturating_add(PAGE - 1) & !(PAGE - 1))
}

/// Add the pages `[a, b)` to `early` (ranges as start -> end), merged with the ranges they overlap
/// or touch.
fn early_add(early: &mut BTreeMap<usize, usize>, mut a: usize, mut b: usize) {
    if a >= b {
        return;
    }
    if let Some((&s, &e)) = early.range(..=a).next_back() {
        if e >= a {
            a = s;
            b = b.max(e);
            early.remove(&s);
        }
    }
    loop {
        let Some((s, e)) = early.range(a..=b).next().map(|(&s, &e)| (s, e)) else { break };
        b = b.max(e);
        early.remove(&s);
    }
    early.insert(a, b);
}

/// Take the pages `[a, b)` out of `early`; a range that overlaps it keeps the rest.
fn early_remove(early: &mut BTreeMap<usize, usize>, a: usize, b: usize) {
    if a >= b {
        return;
    }
    if let Some((&s, &e)) = early.range(..a).next_back() {
        if e > a {
            early.insert(s, a);
            if e > b {
                early.insert(b, e);
                return;
            }
        }
    }
    loop {
        let Some((s, e)) = early.range(a..b).next().map(|(&s, &e)| (s, e)) else { break };
        early.remove(&s);
        if e > b {
            early.insert(b, e);
            break;
        }
    }
}
/// generation counter incremented whenever the set of mapped tiles changes (for watch_ranges cache)
static MAPPED_GEN: AtomicU64 = AtomicU64::new(0);

/// Generation counter of `mapped_ranges()` (changes only when tiles are mapped or unmapped).
pub fn mapped_gen() -> u64 {
    MAPPED_GEN.load(Ordering::Acquire)
}

fn buffer_desc(bytes: u64) -> D3D12_RESOURCE_DESC {
    D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: bytes,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        ..Default::default()
    }
}

unsafe fn new_state(dev: &ID3D12Device, windows: Vec<(usize, ID3D12Resource)>) -> windows::core::Result<State> {
    let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default())?;
    let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE)?;
    let rb_fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE)?;
    let event = windows::Win32::System::Threading::CreateEventW(None, false, false, None)?;
    let rbq = (|| -> windows::core::Result<(ID3D12CommandQueue, ID3D12Fence)> {
        let q: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default())?;
        let _ = q.SetName(windows::core::w!("xcl sparse readback queue"));
        Ok((q, dev.CreateFence(0, D3D12_FENCE_FLAG_NONE)?))
    })()
    .ok();
    let mut slots = Vec::with_capacity(UPLOAD_SLOTS);
    for _ in 0..UPLOAD_SLOTS {
        let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)?;
        let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None)?;
        list.Close()?;
        let _ = list.SetName(windows::core::w!("xcl sparse upload list"));
        let _ = alloc.SetName(windows::core::w!("xcl sparse upload allocator"));
        slots.push(UploadSlot { alloc, list, fence_val: 0 });
    }
    let rb_alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)?;
    let rb_list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &rb_alloc, None)?;
    rb_list.Close()?;
    let _ = rb_list.SetName(windows::core::w!("xcl sparse readback list"));
    let _ = rb_alloc.SetName(windows::core::w!("xcl sparse readback allocator"));
    let _ = queue.SetName(windows::core::w!("xcl sparse queue"));
    Ok(State {
        dev: dev.clone(),
        queue,
        fence,
        value: 0,
        event,
        rb_fence,
        rb_value: 0,
        rb_queue_fences: FxHashMap::default(),
        rbq,
        rbq_value: 0,
        rb_jobs: VecDeque::new(),
        rb_pool: Vec::new(),
        imm_jobs: VecDeque::new(),
        job_seq: 0,
        windows,
        pools: Vec::new(),
        free: Vec::new(),
        retired: Vec::new(),
        mapped: FxHashMap::default(),
        pinned: FxHashMap::default(),
        parked: FxHashMap::default(),
        quarantined: VecDeque::new(),
        queue_fences: FxHashMap::default(),
        kept: FxHashMap::default(),
        pending: Vec::new(),
        buffers: FxHashMap::default(),
        to_zero: BTreeSet::new(),
        staging: None,
        ring_write: 0,
        ring_submitted: 0,
        ring_free: 0,
        inflight: VecDeque::new(),
        zero_buf: None,
        slots,
        next_slot: 0,
        readback: None,
        rb_alloc,
        rb_list,
        rb_last_fence: 0,
        waited: FxHashMap::default(),
        runs: BTreeMap::new(),
        scratch_runs: Vec::new(),
    })
}

/// pool heap number of the tiles of a shadowed window: they are the window's own buffer, not
/// pool tiles (never mapped, freed or pinned)
const SHADOW_POOL: u32 = u32::MAX;

/// A committed window the GPU could not open as a heap (`real::open_uma`): `buf` in video
/// memory stands for `[base, base + size)` and is kept in step like the tiles of a sparse
/// window -- the CPU's writes (write-watch, changed 64-byte chunks) go to it before each
/// ExecuteCommandLists, GPU writes come back on Map and after the lists' memory operations.
/// All of it counts as mapped from the start: the window is committed. False: no queue for
/// the copies (the buffer then never sees the CPU's writes).
pub unsafe fn shadow_window(dev: &ID3D12Device, base: usize, size: usize, buf: &ID3D12Resource) -> bool {
    let mut g = STATE.lock().unwrap();
    if g.is_none() {
        // no sparse window was made (`init`): the copies still need the queue and the lists
        match new_state(dev, Vec::new()) {
            Ok(s) => *g = Some(s),
            Err(e) => {
                crate::xlog_warn!("shadowed window at {base:#x}: {e}");
                return false;
            }
        }
    }
    let Some(s) = g.as_mut() else { return false };
    s.windows.push((base, buf.clone()));
    for (i, t) in (base..base + size).step_by(TILE).enumerate() {
        s.mapped.insert(t, TileEntry::with_holes((SHADOW_POOL, i as u32), 0));
    }
    early_add(&mut s.runs, base, base + size.next_multiple_of(TILE));
    MAPPED_GEN.fetch_add(1, Ordering::Release);
    true
}

/// The sparse windows' buffers on `dev`: `(window_index, buffer_raw, gpu_va)` for each
/// sparse window in `gpu::uma` (empty: no sparse windows).
pub unsafe fn init(dev: &ID3D12Device) -> Vec<(usize, usize, u64)> {
    let Some(u) = crate::gpu::uma::get() else { return Vec::new() };
    let mut out = Vec::new();
    let mut windows = Vec::new();
    for k in 0..u.shader_windows() {
        if !u.is_sparse_window(k) {
            continue;
        }
        let (base, size) = u.window(k);
        let mut r: Option<ID3D12Resource> = None;
        if let Err(e) = dev.CreateReservedResource(&buffer_desc(size as u64), D3D12_RESOURCE_STATE_COMMON, None, &mut r) {
            crate::xlog_warn!("sparse window {k}: CreateReservedResource: {e}");
            return Vec::new();
        }
        let Some(r) = r else { return Vec::new() };
        out.push((k, r.clone().into_raw() as usize, r.GetGPUVirtualAddress()));
        windows.push((base, r));
    }
    if out.is_empty() {
        return out;
    }
    let made = new_state(dev, windows);
    match made {
        Ok(s) => {
            crate::xlog_info!("D3D12.X: {} sparse window(s) for the title's own memory (tiled buffers on video memory)", out.len());
            // a state this replaces (the windows set up again) goes after the lock: the release of
            // its last buffers reaches `forget_buffer`, which takes the lock (#344)
            // (the queue first: `queue()` never answers with one older than the state's)
            let old_queue = QUEUE.lock().unwrap().replace(s.queue.clone());
            let old = STATE.lock().unwrap().replace(s);
            drop(old);
            drop(old_queue);
            for (a, b) in std::mem::take(&mut *EARLY.lock().unwrap()) {
                committed(a, b - a);
            }
            out
        }
        Err(e) => {
            crate::xlog_warn!("sparse windows: {e}");
            Vec::new()
        }
    }
}

impl State {
    #[inline]
    fn window_index(&self, a: usize) -> Option<usize> {
        self.windows.iter().position(|(b, _)| a >= *b && a < *b + crate::gpu::uma::WINDOW)
    }

    unsafe fn take_tile(&mut self) -> Option<((u32, u32), bool)> {
        self.drain_quarantined();
        if self.free.is_empty() && self.flush_parked() {
            self.drain_quarantined();
        }
        if let Some(t) = self.free.pop() {
            return Some((t, true));
        }
        // a pool retired a moment ago (its last tile came free) is taken back before a new
        // 64 MB heap is made: freeing and allocating a tile in turn made a new pool each time,
        // up to 18 GB of retired heaps in 2 s (#913). Its tiles are reused as freed tiles are
        // (zeroed), so this is no less safe than reusing a tile of a pool in use.
        if let Some((heap, _, _)) = self.retired.pop() {
            let h = self.add_pool(heap);
            return Some(((h, 0), true));
        }
        let desc = D3D12_HEAP_DESC {
            SizeInBytes: POOL_TILES as u64 * TILE as u64,
            Properties: D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() },
            Alignment: 0,
            Flags: D3D12_HEAP_FLAG_DENY_RT_DS_TEXTURES | D3D12_HEAP_FLAG_DENY_NON_RT_DS_TEXTURES,
        };
        // (a heap takes its time, with the state locked: what waits for its mapping goes first, a
        // CPU signal that asks meanwhile gives up after 2 ms)
        self.flush_pending();
        let mut heap: Option<ID3D12Heap> = None;
        if let Err(e) = super::title_textures::own_work(|| self.dev.CreateHeap(&desc, &mut heap)) {
            crate::xlog_warn!("sparse windows: no video memory for more tiles ({e})");
            return None;
        }
        let heap = heap?;
        let h = self.add_pool(heap);
        Some(((h, 0), false))
    }

    /// `heap` becomes a pool (in a given-back pool's index if there is one); its tiles but the
    /// first, which the caller takes, go to the free list.
    fn add_pool(&mut self, heap: ID3D12Heap) -> u32 {
        let h = match self.pools.iter().position(|p| p.is_none()) {
            Some(i) => {
                self.pools[i] = Some(heap);
                i as u32
            }
            None => {
                self.pools.push(Some(heap));
                self.pools.len() as u32 - 1
            }
        };
        self.free.extend((1..POOL_TILES).rev().map(|t| (h, t)));
        h
    }

    /// Pools whose every tile is free again go (64 MB of video memory each – and as much of the
    /// system's commit: a retail game's homes of physical pages left 0.75 GB of pools for
    /// 0.29 GB of tiles in use). Call after their tiles are unmapped and signalled.
    fn retire_free_pools(&mut self) {
        let now = std::time::Instant::now();
        let done = unsafe { super::real::fence_completed(&self.fence) };
        self.retired.retain(|(_, v, at)| !(done >= *v && now.duration_since(*at).as_secs() >= 2));
        if self.free.len() < POOL_TILES as usize {
            return;
        }
        let mut count = vec![0u32; self.pools.len()];
        for &(h, _) in &self.free {
            count[h as usize] += 1;
        }
        let gone: Vec<usize> = (0..count.len()).filter(|&h| count[h] == POOL_TILES && self.pools[h].is_some()).collect();
        if gone.is_empty() {
            return;
        }
        self.free.retain(|&(h, _)| !gone.contains(&(h as usize)));
        for h in gone {
            if let Some(heap) = self.pools[h].take() {
                self.retired.push((heap, self.value, now));
            }
        }
    }

    /// Map `tiles` (tile addresses) of `res` (starting at `res_base`) onto pool tiles,
    /// batched per pool heap with contiguous runs merged into single regions.
    unsafe fn map_tiles(&self, res: &ID3D12Resource, res_base: usize, tiles: &[(usize, (u32, u32))]) {
        if tiles.is_empty() {
            return;
        }
        let mut by_heap: FxHashMap<u32, Vec<(u32, u32)>> = FxHashMap::default();
        for &(a, (h, t)) in tiles {
            by_heap.entry(h).or_default().push((((a - res_base) / TILE) as u32, t));
        }
        for (h, mut pairs) in by_heap {
            pairs.sort_unstable_by_key(|&(x, _)| x);
            let mut coords: Vec<D3D12_TILED_RESOURCE_COORDINATE> = Vec::with_capacity(pairs.len());
            let mut regions: Vec<D3D12_TILE_REGION_SIZE> = Vec::with_capacity(pairs.len());
            let mut flags: Vec<D3D12_TILE_RANGE_FLAGS> = Vec::with_capacity(pairs.len());
            let mut offsets: Vec<u32> = Vec::with_capacity(pairs.len());
            let mut counts: Vec<u32> = Vec::with_capacity(pairs.len());
            let mut i = 0;
            while i < pairs.len() {
                let (x0, t0) = pairs[i];
                let mut run = 1u32;
                while i + (run as usize) < pairs.len() {
                    let (nx, nt) = pairs[i + run as usize];
                    if nx == x0 + run && nt == t0 + run {
                        run += 1;
                    } else {
                        break;
                    }
                }
                coords.push(D3D12_TILED_RESOURCE_COORDINATE { X: x0, ..Default::default() });
                regions.push(D3D12_TILE_REGION_SIZE { NumTiles: run, UseBox: false.into(), ..Default::default() });
                flags.push(D3D12_TILE_RANGE_FLAG_NONE);
                offsets.push(t0);
                counts.push(run);
                i += run as usize;
            }
            self.queue.UpdateTileMappings(
                res,
                coords.len() as u32,
                Some(coords.as_ptr()),
                Some(regions.as_ptr()),
                self.pools[h as usize].as_ref().expect("a pool tile of a live pool"),
                flags.len() as u32,
                Some(flags.as_ptr()),
                Some(offsets.as_ptr()),
                Some(counts.as_ptr()),
                D3D12_TILE_MAPPING_FLAG_NONE,
            );
        }
    }

    unsafe fn unmap_tiles(&self, res: &ID3D12Resource, res_base: usize, tiles: &[usize]) {
        if tiles.is_empty() {
            return;
        }
        let mut xs: Vec<u32> = tiles.iter().map(|&a| ((a - res_base) / TILE) as u32).collect();
        xs.sort_unstable();
        let mut coords: Vec<D3D12_TILED_RESOURCE_COORDINATE> = Vec::with_capacity(xs.len());
        let mut regions: Vec<D3D12_TILE_REGION_SIZE> = Vec::with_capacity(xs.len());
        let mut flags: Vec<D3D12_TILE_RANGE_FLAGS> = Vec::with_capacity(xs.len());
        let mut counts: Vec<u32> = Vec::with_capacity(xs.len());
        let mut i = 0;
        while i < xs.len() {
            let x0 = xs[i];
            let mut run = 1u32;
            while i + (run as usize) < xs.len() && xs[i + run as usize] == x0 + run {
                run += 1;
            }
            coords.push(D3D12_TILED_RESOURCE_COORDINATE { X: x0, ..Default::default() });
            regions.push(D3D12_TILE_REGION_SIZE { NumTiles: run, UseBox: false.into(), ..Default::default() });
            flags.push(D3D12_TILE_RANGE_FLAG_NULL);
            counts.push(run);
            i += run as usize;
        }
        self.queue.UpdateTileMappings(
            res,
            coords.len() as u32,
            Some(coords.as_ptr()),
            Some(regions.as_ptr()),
            None,
            flags.len() as u32,
            Some(flags.as_ptr()),
            None,
            Some(counts.as_ptr()),
            D3D12_TILE_MAPPING_FLAG_NONE,
        );
    }

    /// tiles of `[a, a + n)` get video memory (mapped in their window); `initial_holes` supplies
    /// the initial 16-bit hole mask for each newly mapped 64 KiB tile.
    unsafe fn ensure_mapped(&mut self, a: usize, n: usize, mut initial_holes: impl FnMut(usize) -> u16) {
        if n == 0 {
            return;
        }
        let first = a & !(TILE - 1);
        let end = a.saturating_add(n);
        let mut new = false;
        let mut unparked = false;
        let mut t = first;
        while t < end {
            if !self.mapped.contains_key(&t) && self.window_index(t).is_some() {
                // still mapped there: no new mapping, what it held is cleared
                if let Some(tile) = self.parked.remove(&t).or_else(|| self.kept.remove(&t)) {
                    self.mapped.insert(t, TileEntry::with_holes(tile, initial_holes(t)));
                    early_add(&mut self.runs, t, t + TILE);
                    self.to_zero.insert((t, TILE));
                    unparked = true;
                } else {
                    let Some((tile, reused)) = self.take_tile() else { break };
                    self.mapped.insert(t, TileEntry::with_holes(tile, initial_holes(t)));
                    early_add(&mut self.runs, t, t + TILE);
                    if reused {
                        self.to_zero.insert((t, TILE));
                    }
                    self.pending.push(t);
                    new = true;
                }
            }
            match t.checked_add(TILE) {
                Some(next) => t = next,
                None => break,
            }
        }
        if new || unparked {
            MAPPED_GEN.fetch_add(1, Ordering::Release);
        }
        if self.pending.len() >= PENDING_MAX {
            self.flush_pending();
        } else if !self.pending.is_empty() {
            pending_soon();
        }
    }

    /// The pending tiles (`pending`) mapped onto their pool tiles, one call per window, then the
    /// fence signalled (title queues wait for it in `before_execute`). An address decommitted
    /// since has no `mapped` entry and is skipped; one committed again maps its current tile. One
    /// decommitted and parked (or kept for a placed buffer) stays pending: its mapping was never
    /// sent, and a commit that takes it back from there sends none.
    unsafe fn flush_pending(&mut self) {
        if self.pending.is_empty() {
            return;
        }
        let mut addrs = std::mem::take(&mut self.pending);
        addrs.sort_unstable();
        addrs.dedup();
        let mut by_window: FxHashMap<usize, Vec<(usize, (u32, u32))>> = FxHashMap::default();
        for t in addrs {
            if let (Some(e), Some(wi)) = (self.mapped.get(&t), self.window_index(t)) {
                by_window.entry(wi).or_default().push((t, e.pool));
            } else if self.parked.contains_key(&t) || self.kept.contains_key(&t) {
                self.pending.push(t);
            }
        }
        if by_window.is_empty() {
            return;
        }
        for (wi, tiles) in by_window {
            let (b, r) = (self.windows[wi].0, self.windows[wi].1.clone());
            self.map_tiles(&r, b, &tiles);
        }
        self.signal();
    }

    #[inline]
    unsafe fn signal(&mut self) {
        self.value += 1;
        let _ = self.queue.Signal(&self.fence, self.value);
    }

    #[inline]
    unsafe fn wait_for(&self, target: u64) {
        if target == 0 || super::real::fence_completed(&self.fence) >= target {
            return;
        }
        // The event is one for all waits here, and a wait that gave up leaves its registration
        // behind: that one sets the event later, and the next wait would return at once with the
        // queue not there (10 such "after 5 s" lines within 100 ms in one run, each followed by a
        // Reset of an allocator the GPU was still in). So: until the fence says so, or 5 s.
        let t0 = std::time::Instant::now();
        loop {
            let _ = self.fence.SetEventOnCompletion(target, self.event);
            let left = 5000u64.saturating_sub(t0.elapsed().as_millis() as u64).max(1) as u32;
            windows::Win32::System::Threading::WaitForSingleObject(self.event, left);
            if super::real::fence_completed(&self.fence) >= target {
                return;
            }
            if t0.elapsed().as_millis() >= 5000 {
                break;
            }
        }
        crate::xlog_warn!("sparse: the queue has not reached fence value {target} after 5 s (at {})", super::real::fence_completed(&self.fence));
    }

    #[inline]
    unsafe fn reclaim_ring(&mut self) {
        if self.inflight.is_empty() {
            return;
        }
        let completed = super::real::fence_completed(&self.fence);
        while let Some(&(fv, end_pos)) = self.inflight.front() {
            if fv <= completed {
                self.ring_free = end_pos;
                self.inflight.pop_front();
            } else {
                break;
            }
        }
    }

    /// Acquire the next upload command slot, waiting only if that slot's previous submission is
    /// still executing on the GPU, then reset its allocator and list.
    unsafe fn begin_slot(&mut self) -> usize {
        let idx = self.next_slot;
        self.next_slot = (self.next_slot + 1) % self.slots.len();
        let fv = self.slots[idx].fence_val;
        if fv > 0 && super::real::fence_completed(&self.fence) < fv {
            self.wait_for(fv);
            self.reclaim_ring();
        }
        let slot = &self.slots[idx];
        let _ = slot.alloc.Reset();
        let _ = slot.list.Reset(&slot.alloc, None);
        idx
    }

    /// Close and submit upload slot `idx` asynchronously (NO CPU wait — callers order GPU work
    /// via `queue.Wait(&self.fence, self.value)`).
    unsafe fn submit_slot(&mut self, idx: usize) {
        // the copies on `queue` see every tile mapped (`pending` goes first on the same queue)
        self.flush_pending();
        let closed = self.slots[idx].list.Close().is_ok();
        if closed {
            if let Ok(l) = self.slots[idx].list.cast::<ID3D12CommandList>() {
                super::title_textures::own_work(|| {
                    self.queue.ExecuteCommandLists(&[Some(l)]);
                });
            }
        }
        self.signal();
        self.slots[idx].fence_val = self.value;
        if self.ring_write > self.ring_submitted {
            self.inflight.push_back((self.value, self.ring_write));
            self.ring_submitted = self.ring_write;
        }
    }

    unsafe fn staging_ptr(&mut self) -> Option<*mut u8> {
        if self.staging.is_none() {
            let mut r: Option<ID3D12Resource> = None;
            let mut d = buffer_desc(STAGING as u64);
            d.Flags = D3D12_RESOURCE_FLAG_NONE;
            self.dev
                .CreateCommittedResource(
                    &D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_UPLOAD, ..Default::default() },
                    D3D12_HEAP_FLAG_NONE,
                    &d,
                    D3D12_RESOURCE_STATE_GENERIC_READ,
                    None,
                    &mut r,
                )
                .ok()?;
            let r = r?;
            let mut p = std::ptr::null_mut();
            r.Map(0, None, Some(&mut p)).ok()?;
            self.staging = Some((r, p as *mut u8));
        }
        self.staging.as_ref().map(|&(_, p)| p)
    }

    unsafe fn ensure_zero_buf(&mut self) -> bool {
        if self.zero_buf.is_none() {
            let mut r: Option<ID3D12Resource> = None;
            let mut d = buffer_desc(ZERO_BUF as u64);
            d.Flags = D3D12_RESOURCE_FLAG_NONE;
            if self
                .dev
                .CreateCommittedResource(
                    &D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_UPLOAD, ..Default::default() },
                    D3D12_HEAP_FLAG_NONE,
                    &d,
                    D3D12_RESOURCE_STATE_GENERIC_READ,
                    None,
                    &mut r,
                )
                .is_err()
            {
                return false;
            }
            let Some(r) = r else { return false };
            let mut p = std::ptr::null_mut();
            if r.Map(0, None, Some(&mut p)).is_err() {
                return false;
            }
            std::ptr::write_bytes(p as *mut u8, 0, ZERO_BUF);
            r.Unmap(0, None);
            self.zero_buf = Some(r);
        }
        true
    }

    unsafe fn readback_buf(&mut self) -> Option<(ID3D12Resource, *mut u8)> {
        if self.readback.is_none() {
            let mut r: Option<ID3D12Resource> = None;
            let mut d = buffer_desc(READBACK_STAGING as u64);
            d.Flags = D3D12_RESOURCE_FLAG_NONE;
            self.dev
                .CreateCommittedResource(
                    &D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() },
                    D3D12_HEAP_FLAG_NONE,
                    &d,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    None,
                    &mut r,
                )
                .ok()?;
            let r = r?;
            let mut p = std::ptr::null_mut();
            r.Map(0, None, Some(&mut p)).ok()?;
            self.readback = Some((r, p as *mut u8));
        }
        self.readback.clone()
    }

    /// Clear `sent` hashes for every page overlapping `[a, a + n)`.
    fn clear_sent_range(&mut self, a: usize, n: usize) {
        let mut pg = a & !(PAGE - 1);
        let end = a + n;
        while pg < end {
            let t = pg & !(TILE - 1);
            let tile_end = (t + TILE).min(end);
            if let Some(entry) = self.mapped.get_mut(&t) {
                let mut p = pg;
                while p < tile_end {
                    entry.sent[(p - t) / PAGE] = None;
                    p += PAGE;
                }
            }
            pg = tile_end;
        }
    }

    /// Update `sent` hashes for all 64-byte chunks overlapping `[va, va + len)` so the CPU
    /// copy-back we just wrote to `va` is not re-uploaded by the next `before_execute`.
    unsafe fn sync_sent_range(&mut self, va: usize, len: usize) {
        if len == 0 {
            return;
        }
        let mut pg = va & !(PAGE - 1);
        let end = va + len;
        while pg < end {
            let t = pg & !(TILE - 1);
            let tile_end = (t + TILE).min(end);
            if let Some(entry) = self.mapped.get_mut(&t) {
                let mut p = pg;
                while p < tile_end {
                    let idx = (p - t) / PAGE;
                    if entry.holes & (1u16 << idx) == 0 {
                        let bytes = std::slice::from_raw_parts(p as *const u8, PAGE);
                        let sent = entry.sent[idx].get_or_insert_with(|| Box::new(ZERO_HASHES));
                        let lo = va.saturating_sub(p) / CHUNK;
                        let hi = (end.min(p + PAGE) - p).div_ceil(CHUNK).min(CHUNKS);
                        for k in lo..hi {
                            sent[k] = chunk_hash(&bytes[k * CHUNK..(k + 1) * CHUNK]);
                        }
                    }
                    p += PAGE;
                }
            }
            pg = tile_end;
        }
    }

    /// Whether nobody wrote the 64-byte chunk holding `va` since it was last sent to the GPU (its
    /// page has video memory and its hash is the one known).
    unsafe fn chunk_clean(&self, va: usize) -> bool {
        let pg = va & !(PAGE - 1);
        let t = pg & !(TILE - 1);
        let idx = (pg - t) / PAGE;
        let Some(entry) = self.mapped.get(&t) else { return false };
        if entry.holes & (1u16 << idx) != 0 {
            return false;
        }
        let page = std::slice::from_raw_parts(pg as *const u8, PAGE);
        let k = (va - pg) / CHUNK;
        let h = chunk_hash(&page[k * CHUNK..(k + 1) * CHUNK]);
        entry.sent[idx].as_deref().map_or(h == ZERO_HASH, |sent| h == sent[k])
    }

    /// `len` bytes from readback memory `p + off` into the title's `dst` (committed pages only)
    unsafe fn copy_back(&mut self, p: *mut u8, dst: usize, off: usize, len: usize) {
        let mut at = dst;
        let mut src_off = off;
        let end = dst + len;
        while at < end {
            let pg = at & !(PAGE - 1);
            let t = pg & !(TILE - 1);
            let idx = (pg - t) / PAGE;
            let step = (pg + PAGE).min(end) - at;
            if let Some(entry) = self.mapped.get_mut(&t) {
                if entry.holes & (1u16 << idx) == 0 {
                    land_page(&mut entry.sent[idx], pg, at - pg, p.add(src_off), step);
                }
            }
            at += step;
            src_off += step;
        }
    }

    /// An immediate the GPU wrote at `va` (`bytes`: 4 or 8), into the title's memory there
    unsafe fn store_imm(&mut self, va: usize, value: u64, bytes: u8) {
        let pg = va & !(PAGE - 1);
        let t = pg & !(TILE - 1);
        if !self.mapped.get(&t).is_some_and(|e| e.holes & (1u16 << ((pg - t) / PAGE)) == 0) {
            return;
        }
        // (a chunk the title's CPU wrote since it was last sent keeps its state: it goes up with the
        // next look, with this value in it)
        let clean = self.chunk_clean(va);
        match bytes & 0x7f {
            4 => (*(va as *const std::sync::atomic::AtomicU32)).store(value as u32, Ordering::Release),
            8 if bytes & IMM_KEEP_MAX != 0 => {
                if (*(va as *const AtomicU64)).fetch_max(value, Ordering::AcqRel) > value {
                    static LATE: AtomicU64 = AtomicU64::new(0);
                    let k = LATE.fetch_add(1, Ordering::Relaxed) + 1;
                    if k.is_power_of_two() {
                        crate::xlog_info!("sparse: {k} immediates so far left out – their cell is a counter that held a later value ({value:#x} for {va:#x})");
                    }
                    return;
                }
            }
            8 => (*(va as *const AtomicU64)).store(value, Ordering::Release),
            _ => return,
        }
        if clean {
            self.sync_sent_range(va, (bytes & 0x7f) as usize);
        }
    }

    /// A copy back's list and a readback buffer of at least `need` bytes (reused when one fits)
    unsafe fn rb_job(&mut self, need: usize, ty: D3D12_COMMAND_LIST_TYPE) -> Option<RbJob> {
        if let Some(i) = self.rb_pool.iter().enumerate().filter(|(_, j)| j.cap >= need && j.ty == ty).min_by_key(|(_, j)| j.cap).map(|(i, _)| i) {
            return Some(self.rb_pool.swap_remove(i));
        }
        let cap = need.max(64 << 10).next_power_of_two().min(READBACK_STAGING);
        let alloc: ID3D12CommandAllocator = self.dev.CreateCommandAllocator(ty).ok()?;
        let list: ID3D12GraphicsCommandList = self.dev.CreateCommandList(0, ty, &alloc, None).ok()?;
        list.Close().ok()?;
        let _ = list.SetName(windows::core::w!("xcl sparse readback list"));
        let mut d = buffer_desc(cap as u64);
        d.Flags = D3D12_RESOURCE_FLAG_NONE;
        let mut r: Option<ID3D12Resource> = None;
        self.dev
            .CreateCommittedResource(&D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() }, D3D12_HEAP_FLAG_NONE, &d, D3D12_RESOURCE_STATE_COPY_DEST, None, &mut r)
            .ok()?;
        let buf = r?;
        let mut p: *mut c_void = std::ptr::null_mut();
        buf.Map(0, None, Some(&mut p)).ok()?;
        Some(RbJob { alloc, list, buf, ptr: p as *mut u8, cap, rb_v: 0, done_v: 0, copies: Vec::new(), ty, fences: None, seq: 0 })
    }

    /// Land submitted copies back in the title's memory as far as `how` says: every job that is
    /// done, a queue's own in their order (a job of a queue that waits does not hold back the
    /// other queues' jobs).
    unsafe fn drain_readbacks(&mut self, how: Drain) {
        self.drain_readbacks_spin(how, 2000);
    }

    /// `drain_readbacks`, waiting for a finished list's copy for `spin_us` at most.
    unsafe fn drain_readbacks_spin(&mut self, how: Drain, spin_us: u128) {
        let rbq_done = self.rbq.as_ref().map(|(_, f)| f.clone());
        let mut i = 0;
        let mut k = 0;
        loop {
            // the immediates of submissions before the next copy back first
            while k < self.imm_jobs.len() && self.rb_jobs.get(i).is_none_or(|j| self.imm_jobs[k].seq < j.seq) {
                if super::real::fence_completed(&self.imm_jobs[k].fence) < self.imm_jobs[k].value {
                    k += 1;
                    continue;
                }
                let Some(job) = self.imm_jobs.remove(k) else { break };
                for &(va, value, bytes) in &job.writes {
                    self.store_imm(va, value, bytes);
                }
                RB_PENDING.fetch_sub(1, Ordering::Release);
            }
            if i >= self.rb_jobs.len() {
                break;
            }
            let (title_f, done_f, rb_v, done_v) = {
                let j = &self.rb_jobs[i];
                match (&j.fences, &rbq_done) {
                    (Some((a, b)), _) => (a.clone(), b.clone(), j.rb_v, j.done_v),
                    (None, Some(d)) => (self.rb_fence.clone(), d.clone(), j.rb_v, j.done_v),
                    (None, None) => {
                        i += 1;
                        continue;
                    }
                }
            };
            if super::real::fence_completed(&done_f) < done_v {
                if how == Drain::Finished || super::real::fence_completed(&title_f) < rb_v {
                    i += 1;
                    continue;
                }
                // the title's lists are done: only the copy itself is left
                // (for 2 ms at most: the state's lock is held here, and every submit needs it; a copy
                // that stands behind something else on its queue lands when it is done)
                // (no event: `self.event` is the tile queue's, see `wait_for`)
                let t0 = std::time::Instant::now();
                while super::real::fence_completed(&done_f) < done_v && t0.elapsed().as_micros() < spin_us {
                    std::thread::yield_now();
                }
                if super::real::fence_completed(&done_f) < done_v {
                    static LATE: AtomicU64 = AtomicU64::new(0);
                    let k = LATE.fetch_add(1, Ordering::Relaxed) + 1;
                    if k.is_power_of_two() {
                        crate::xlog_warn!("sparse: {k} copies back so far not finished 2 ms after the title's work they follow – left for later");
                    }
                    i += 1;
                    continue;
                }
            }
            let Some(mut job) = self.rb_jobs.remove(i) else { break };
            let copies = std::mem::take(&mut job.copies);
            for &(dst, off, len) in &copies {
                self.copy_back(job.ptr, dst, off, len);
            }
            RB_PENDING.fetch_sub(1, Ordering::Release);
            job.copies = copies;
            job.copies.clear();
            // (a burst of copies back does not keep its buffers for good: each is up to 16 MiB of
            // host memory)
            if self.rb_pool.len() < 32 {
                self.rb_pool.push(job);
            }
        }
    }

    /// `readback_ranges` without a CPU wait: the copies are recorded on the title's own `queue`,
    /// right behind the lists that wrote `ranges` (so they wait for those lists and for nothing
    /// else), and land in the title's memory in `drain_readbacks` once `fences.1` reaches `v`.
    unsafe fn readback_async(&mut self, queue: &ID3D12CommandQueue, ranges: &[(usize, usize)], fences: &(ID3D12Fence, ID3D12Fence), v: u64) {
        let ty = queue.GetDesc().Type;
        // the mapped parts, in readback-buffer sized batches: (window, address, bytes)
        let mut batches: Vec<Vec<(usize, usize, usize)>> = vec![Vec::new()];
        let mut used = 0usize;
        for &(a, n) in ranges {
            let mut at = 0usize;
            while at < n {
                let cur = a + at;
                let tile = cur & !(TILE - 1);
                let Some(wi) = self.window_index(cur) else { break };
                let to_tile_end = (tile + TILE) - cur;
                if !self.mapped.contains_key(&tile) {
                    at += (n - at).min(to_tile_end);
                    continue;
                }
                if used == READBACK_STAGING {
                    batches.push(Vec::new());
                    used = 0;
                }
                let to_win_end = (self.windows[wi].0 + crate::gpu::uma::WINDOW) - cur;
                let chunk = (n - at).min(READBACK_STAGING - used).min(to_tile_end).min(to_win_end);
                batches.last_mut().expect("batch").push((wi, cur, chunk));
                used += chunk;
                at += chunk;
            }
        }
        for batch in batches.into_iter().filter(|b| !b.is_empty()) {
            let need = batch.iter().map(|c| c.2).sum::<usize>();
            let Some(mut job) = self.rb_job(need, ty) else {
                crate::xlog_warn!("sparse: no readback buffer for a copy back of {need} bytes – left out");
                // (the Signal after the loop still goes out: what was recorded for it stands)
                break;
            };
            let _ = job.alloc.Reset();
            let _ = job.list.Reset(&job.alloc, None);
            let mut off = 0usize;
            for (wi, cur, chunk) in batch {
                let (base, res) = (self.windows[wi].0, self.windows[wi].1.clone());
                job.list.CopyBufferRegion(&job.buf, off as u64, &res, (cur - base) as u64, chunk as u64);
                job.copies.push((cur, off, chunk));
                off += chunk;
            }
            if job.list.Close().is_ok() {
                pc_execute(queue, &job.list);
            }
            job.rb_v = v;
            job.done_v = v;
            job.fences = Some(fences.clone());
            self.job_seq += 1;
            job.seq = self.job_seq;
            self.rb_jobs.push_back(job);
            RB_PENDING.fetch_add(1, Ordering::Release);
            RB_ASYNC.store(true, Ordering::Relaxed);
        }
        let _ = queue.Signal(&fences.1, v);
    }

    /// Copy `ranges` (address, bytes) from sparse VRAM back into the CPU view and refresh
    /// `sent` hashes. Caller has already ordered `self.queue` after any GPU writes if needed.
    unsafe fn readback_ranges(&mut self, ranges: &[(usize, usize)]) {
        self.flush_pending();
        // copies back of earlier submissions first: their data is older than this one's
        self.drain_readbacks(Drain::TitleDone);
        let Some((rb, p)) = self.readback_buf() else { return };
        let mut pending: Vec<(usize, usize, usize)> = Vec::new(); // (va, rb_off, chunk)
        let mut used = 0usize;
        let mut open = false;
        let flush_and_copy = |s: &mut State, open: &mut bool, pending: &mut Vec<(usize, usize, usize)>, used: &mut usize, p: *mut u8| {
            if *open {
                *open = false;
                if s.rb_list.Close().is_ok() {
                    if let Ok(l) = s.rb_list.cast::<ID3D12CommandList>() {
                        super::title_textures::own_work(|| {
                            s.queue.ExecuteCommandLists(&[Some(l)]);
                        });
                    }
                }
                s.signal();
                s.rb_last_fence = s.value;
                s.wait_for(s.value);
                s.reclaim_ring();
            }
            for &(dst, off, len) in pending.iter() {
                s.copy_back(p, dst, off, len);
            }
            pending.clear();
            *used = 0;
        };
        for &(a, n) in ranges {
            if n == 0 {
                continue;
            }
            let mut done = 0usize;
            while done < n {
                let cur = a + done;
                let tile = cur & !(TILE - 1);
                let Some(wi) = self.window_index(cur) else { break };
                let b = self.windows[wi].0;
                let to_tile_end = (tile + TILE) - cur;
                if !self.mapped.contains_key(&tile) {
                    done += (n - done).min(to_tile_end);
                    continue;
                }
                if used == READBACK_STAGING {
                    flush_and_copy(self, &mut open, &mut pending, &mut used, p);
                }
                let to_win_end = (b + crate::gpu::uma::WINDOW) - cur;
                let chunk = (n - done).min(READBACK_STAGING - used).min(to_tile_end).min(to_win_end);
                if !open {
                    self.wait_for(self.rb_last_fence);
                    let _ = self.rb_alloc.Reset();
                    let _ = self.rb_list.Reset(&self.rb_alloc, None);
                    open = true;
                }
                self.rb_list.CopyBufferRegion(&rb, used as u64, &self.windows[wi].1, (cur - b) as u64, chunk as u64);
                pending.push((cur, used, chunk));
                used += chunk;
                done += chunk;
            }
        }
        if open {
            flush_and_copy(self, &mut open, &mut pending, &mut used, p);
        }
    }

    /// Asynchronously zero `runs` (address, bytes) in sparse VRAM using the pre-zeroed `zero_buf`
    /// (zero staging ring bytes consumed, zero CPU stall).
    unsafe fn upload_zeros(&mut self, runs: &[(usize, usize)]) {
        if runs.is_empty() || !self.ensure_zero_buf() {
            return;
        }
        let slot_idx = self.begin_slot();
        let mut last_wi: Option<usize> = None;
        let zero_buf = self.zero_buf.as_ref().unwrap();
        let list = &self.slots[slot_idx].list;
        for &(a, n) in runs {
            let mut done = 0usize;
            while done < n {
                let cur = a + done;
                let wi = match last_wi {
                    Some(i) if cur >= self.windows[i].0 && cur < self.windows[i].0 + crate::gpu::uma::WINDOW => i,
                    _ => match self.window_index(cur) {
                        Some(i) => {
                            last_wi = Some(i);
                            i
                        }
                        None => break,
                    },
                };
                let b = self.windows[wi].0;
                let to_win_end = (b + crate::gpu::uma::WINDOW) - cur;
                let chunk = (n - done).min(ZERO_BUF).min(to_win_end);
                list.CopyBufferRegion(&self.windows[wi].1, (cur - b) as u64, zero_buf, 0, chunk as u64);
                done += chunk;
            }
        }
        self.submit_slot(slot_idx);
    }

    /// Asynchronously copy `runs` (address, bytes) from the CPU view into sparse VRAM tiles via
    /// the 64 MiB ring staging buffer (zero CPU stall on the common path).
    unsafe fn upload_dirty(&mut self, runs: &[(usize, usize)]) {
        if runs.is_empty() {
            return;
        }
        let Some(p) = self.staging_ptr() else { return };
        let mut open_slot: Option<usize> = None;
        let mut last_wi: Option<usize> = None;
        // the copies into the ring (source address, ring offset, bytes), done before their slot is
        // submitted, by the worker threads when there are many
        let mut jobs: Vec<(usize, usize, usize)> = Vec::new();
        // the many small ones first, together (the windows the open slot's dispatches write are
        // UAVs in it until their first copy)
        let mut uavs: Vec<usize> = Vec::new();
        let single = self.scatter_small(runs, p, &mut open_slot, &mut jobs, &mut uavs);
        for &(a, n) in single.as_deref().unwrap_or(runs) {
            let mut done = 0usize;
            while done < n {
                let cur = a + done;
                let wi = match last_wi {
                    Some(i) if cur >= self.windows[i].0 && cur < self.windows[i].0 + crate::gpu::uma::WINDOW => i,
                    _ => match self.window_index(cur) {
                        Some(i) => {
                            last_wi = Some(i);
                            i
                        }
                        None => break,
                    },
                };
                let b = self.windows[wi].0;
                let to_win_end = (b + crate::gpu::uma::WINDOW) - cur;
                let (aligned, chunk) = plan_ring_chunk(self.ring_write, (n - done).min(to_win_end), STAGING);
                self.ring_room(aligned + chunk as u64, p, &mut open_slot, &mut jobs);
                let slot_idx = match open_slot {
                    Some(idx) => idx,
                    None => {
                        uavs.clear();
                        let idx = self.begin_slot();
                        open_slot = Some(idx);
                        idx
                    }
                };
                if let Some(k) = uavs.iter().position(|&w| w == wi) {
                    uavs.swap_remove(k);
                    self.slots[slot_idx].list.ResourceBarrier(&[uav_to_copy(&self.windows[wi].1)]);
                }
                let dst_off = (aligned as usize) % STAGING;
                jobs.push((cur, dst_off, chunk));
                let staging_res = &self.staging.as_ref().unwrap().0;
                self.slots[slot_idx].list.CopyBufferRegion(&self.windows[wi].1, (cur - b) as u64, staging_res, dst_off as u64, chunk as u64);
                self.ring_write = aligned + chunk as u64;
                done += chunk;
            }
        }
        if let Some(slot_idx) = open_slot {
            run_copies(&mut jobs, p);
            self.submit_slot(slot_idx);
        }
    }

    /// Room in the upload ring up to its position `end`: what the GPU is done with is taken back;
    /// when that is not enough, the open slot is submitted and the GPU waited for.
    unsafe fn ring_room(&mut self, end: u64, p: *mut u8, open_slot: &mut Option<usize>, jobs: &mut Vec<(usize, usize, usize)>) {
        if end.saturating_sub(self.ring_free) <= STAGING as u64 {
            return;
        }
        self.reclaim_ring();
        if end.saturating_sub(self.ring_free) <= STAGING as u64 {
            return;
        }
        if let Some(slot_idx) = open_slot.take() {
            run_copies(jobs, p);
            self.submit_slot(slot_idx);
            self.reclaim_ring();
        }
        while end.saturating_sub(self.ring_free) > STAGING as u64 {
            let Some(&(fv, _)) = self.inflight.front() else { break };
            self.wait_for(fv);
            self.reclaim_ring();
        }
    }

    /// Record the small ones of `runs` as dispatches (`scatter`) that put many of them in their
    /// places at once, in the open slot: the runs left for a copy each (None: all of them). `uavs`
    /// are the windows the open slot's dispatches write.
    unsafe fn scatter_small(&mut self, runs: &[(usize, usize)], p: *mut u8, open_slot: &mut Option<usize>, jobs: &mut Vec<(usize, usize, usize)>, uavs: &mut Vec<usize>) -> Option<Vec<(usize, usize)>> {
        if runs.len() < SCATTER_FROM || !super::scatter::on() {
            return None;
        }
        // (a run that ends in the next window is cut where the window ends, by the copies)
        let small = |&(a, n): &(usize, usize)| n < SCATTER_RUN && a % CHUNK == 0 && n % CHUNK == 0 && self.window_index(a).is_some_and(|w| a + n <= self.windows[w].0 + crate::gpu::uma::WINDOW);
        let (small, single): (Vec<_>, Vec<_>) = runs.iter().copied().partition(small);
        if small.len() < SCATTER_FROM {
            return None;
        }
        let pass = super::scatter::of(&self.dev)?;
        let ring_va = self.staging.as_ref()?.0.GetGPUVirtualAddress();
        let pieces = super::scatter::pieces(&small, SCATTER_CHUNKS, |a| self.window_index(a).map(|w| (w, self.windows[w].0)));
        let mut bound = None;
        for piece in &pieces {
            // the table of the piece and its chunks in one part of the ring
            let bytes = super::scatter::Scatter::bytes(piece.chunks);
            let mut at = self.ring_write.next_multiple_of(RING_ALIGN as u64);
            let to_end = STAGING - at as usize % STAGING;
            if bytes > to_end {
                at += to_end as u64;
            }
            self.ring_room(at + bytes as u64, p, open_slot, jobs);
            let slot_idx = match *open_slot {
                Some(idx) => idx,
                None => {
                    uavs.clear();
                    let idx = self.begin_slot();
                    *open_slot = Some(idx);
                    idx
                }
            };
            let list = self.slots[slot_idx].list.clone();
            if bound != Some((slot_idx, self.slots[slot_idx].fence_val)) {
                bound = Some((slot_idx, self.slots[slot_idx].fence_val));
                pass.bind(&list);
            }
            let off = at as usize % STAGING;
            let mut place = p.add(off) as *mut u32;
            let mut data = off + super::scatter::Scatter::data_at(piece.chunks);
            for &(a, n) in &small[piece.runs.clone()] {
                for k in 0..n / CHUNK {
                    place.write((a - piece.base + k * CHUNK) as u32);
                    place = place.add(1);
                }
                jobs.push((a, data, n));
                data += n;
            }
            let (base, buf) = &self.windows[piece.window];
            pass.dispatch(&list, piece.chunks, ring_va + off as u64, buf.GetGPUVirtualAddress() + (piece.base - base) as u64);
            if !uavs.contains(&piece.window) {
                uavs.push(piece.window);
            }
            self.ring_write = at + bytes as u64;
            SCATTERED[0].fetch_add(1, Ordering::Relaxed);
        }
        SCATTERED[1].fetch_add(small.len() as u64, Ordering::Relaxed);
        Some(single)
    }
}

/// dispatches of `scatter_small` so far, and the runs in them
static SCATTERED: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// `r`, written by a dispatch of the list, becomes the destination of a copy.
fn uav_to_copy(r: &ID3D12Resource) -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                // SAFETY: the list keeps no reference and the window outlives it
                pResource: unsafe { std::mem::transmute_copy(r) },
                Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                StateBefore: D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                StateAfter: D3D12_RESOURCE_STATE_COPY_DEST,
            }),
        },
    }
}

/// Allocate a zeroed, write-watched block in the sparse VRAM windows (never straddling a
/// 4 GiB window boundary when `size <= 4 GiB`). Commits the CPU pages and maps VRAM tiles
/// (or queues them until the D3D12 device exists). Returns `(address, rounded_size)`.
pub unsafe fn alloc(size: usize, align: usize) -> Option<(usize, usize)> {
    let u = crate::gpu::uma::get()?;
    let (addr, rounded) = u.sparse_alloc(size, align)?;
    if VirtualAlloc(addr as *const c_void, rounded, MEM_COMMIT, PAGE_READWRITE).is_null() {
        u.sparse_release(addr);
        return None;
    }
    committed(addr, rounded);
    Some((addr, rounded))
}

/// Allocate a zeroed, write-watched block strictly inside sparse window `window` (e.g.
/// window 0 for 32-bit-addressed descriptor heaps). Returns `(address, rounded_size)`.
pub unsafe fn alloc_in(window: usize, size: usize, align: usize) -> Option<(usize, usize)> {
    let u = crate::gpu::uma::get()?;
    let (addr, rounded) = u.sparse_alloc_in(window, size, align)?;
    if VirtualAlloc(addr as *const c_void, rounded, MEM_COMMIT, PAGE_READWRITE).is_null() {
        u.sparse_release(addr);
        return None;
    }
    committed(addr, rounded);
    Some((addr, rounded))
}

/// Release a sparse allocation from `alloc` / `alloc_in` (false: not a sparse allocation).
pub unsafe fn free(p: usize) -> bool {
    let Some(u) = crate::gpu::uma::get() else { return false };
    if !u.is_sparse(p) {
        return false;
    }
    // decommitted before the address goes back: once released, another thread may get it,
    // commit it and write there
    let Some((start, len)) = u.sparse_reservation(p).filter(|&(start, _)| start == p) else { return false };
    decommitted(start, len);
    VirtualFree(p as *mut c_void, len, MEM_DECOMMIT);
    u.sparse_release(p).is_some()
}

/// After `queue.ExecuteCommandLists`: copy GPU-written sparse ranges (e.g. `Write32/64`,
/// `FillMemoryWith32BitValueX`, `CopyMemoryToMemoryX`) back to their CPU view so CPU reads
/// and spin-waits see the GPU writes without clobbering `sent` hashes.
/// The queue tile mappings go out on, once there is one. Answered without `STATE` (`QUEUE`):
/// a caller may be inside a D3D12 call that a holder of `STATE` waits for.
pub fn queue() -> Option<ID3D12CommandQueue> {
    QUEUE.lock().unwrap().clone()
}

pub unsafe fn readback_after_execute(queue: &ID3D12CommandQueue, ranges: &[(usize, usize)], imms: &[(usize, u64, u8)]) {
    if ranges.is_empty() && imms.is_empty() {
        return;
    }
    let mut g = lock_state();
    let Some(s) = g.as_mut() else { return };
    s.note_queue(queue);
    s.after_execute(queue, ranges, imms);
}

/// The PC `ExecuteCommandLists` of `queue` for `list` (an adopted title queue's own method,
/// not XCL's hook: this is not a submit of the title's).
pub(super) unsafe fn pc_execute(queue: &ID3D12CommandQueue, list: &ID3D12GraphicsCommandList) {
    type ExecFn = unsafe extern "system" fn(*mut c_void, u32, *const *mut c_void);
    const SLOT: usize = 10;
    let q = queue.as_raw();
    let f: ExecFn = if super::real::is_adopted(q) { std::mem::transmute(super::real::orig_fn(q, SLOT)) } else { std::mem::transmute(*(*(q as *const *const usize)).add(SLOT)) };
    let one = [list.as_raw()];
    f(q, 1, one.as_ptr());
}

/// copies back submitted and not landed yet, and whether any ran on the readback queue
static RB_PENDING: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
static RB_ASYNC: AtomicBool = AtomicBool::new(false);

/// Whether copies back land asynchronously (any ran on the readback queue so far)
pub fn readbacks_async() -> bool {
    RB_ASYNC.load(Ordering::Relaxed)
}

/// Copies back that are done on the GPU, into the title's memory, and those whose title work
/// is done (waiting for the copy): before a CPU read of memory the title's lists wrote.
pub unsafe fn settle_readbacks() {
    if RB_PENDING.load(Ordering::Acquire) == 0 {
        return;
    }
    if let Some(s) = STATE.lock().unwrap().as_mut() {
        s.drain_readbacks(Drain::TitleDone);
    }
}

/// `settle_readbacks` from a fence observation: the state's lock is only tried (for ~2 ms), as
/// XCL may be waiting on a fence of its own with the lock held.
pub unsafe fn settle_readbacks_try() {
    settle_readbacks_within(5);
}

/// `settle_readbacks` with the state's lock waited for `ms` ms at most (false: it was not had:
/// what is pending stays pending). A copy back whose title work is done is waited for 0.3 ms.
pub unsafe fn settle_readbacks_within(ms: u64) -> bool {
    if RB_PENDING.load(Ordering::Acquire) == 0 {
        return true;
    }
    let t0 = std::time::Instant::now();
    loop {
        if let Ok(mut g) = STATE.try_lock() {
            if let Some(s) = g.as_mut() {
                s.drain_readbacks_spin(Drain::TitleDone, 300);
            }
            return true;
        }
        if t0.elapsed().as_millis() as u64 >= ms {
            return false;
        }
        std::thread::yield_now();
    }
}

/// The thread that moves finished copies back into the title's memory, so it sees what its
/// lists wrote without submitting again (started with the first copy back).
fn start_readback_thread() {
    static STARTED: std::sync::Once = std::sync::Once::new();
    STARTED.call_once(|| {
        let _ = std::thread::Builder::new().name("xcl-sparse-readback".into()).spawn(|| loop {
            std::thread::sleep(std::time::Duration::from_micros(500));
            if let Ok(mut g) = STATE.try_lock() {
                if let Some(s) = g.as_mut().filter(|s| !s.rb_jobs.is_empty() || !s.imm_jobs.is_empty()) {
                    unsafe { s.drain_readbacks(Drain::Finished) };
                }
            }
        });
    });
}

impl State {
    /// `readback_after_execute` on this state
    unsafe fn after_execute(&mut self, queue: &ID3D12CommandQueue, ranges: &[(usize, usize)], imms: &[(usize, u64, u8)]) {
        let s = self;
        let mut filtered: Vec<(usize, usize)> = ranges.iter().copied().filter(|&(a, n)| n > 0 && s.window_index(a).is_some()).collect();
        // immediates: stored by XCL when the lists are done. One that another write of these
        // lists overlaps, and all of them without a queue of the title's to order them on,
        // come back with a copy like the rest.
        let mut imm: Vec<(usize, u64, u8)> = Vec::new();
        for &(a, v, n) in imms {
            if s.window_index(a).is_none() {
                continue;
            }
            let len = (n & 0x7f) as usize;
            if s.rbq.is_none() || ranges.iter().any(|&(ra, rn)| a < ra + rn && ra < a + len) {
                filtered.push((a, len));
            } else {
                imm.push((a, v, n));
            }
        }
        if filtered.is_empty() && imm.is_empty() {
            return;
        }
        filtered.sort_unstable();
        let mut merged: Vec<(usize, usize)> = Vec::with_capacity(filtered.len());
        for (a, n) in filtered {
            match merged.last_mut() {
                Some(last) if a <= last.0.saturating_add(last.1) => {
                    let end = (last.0 + last.1).max(a + n);
                    last.1 = end - last.0;
                }
                _ => merged.push((a, n)),
            }
        }
        if s.rbq.is_some() {
            // on this queue: [the title's lists] Signal(first, v) [the copies] Signal(second, v)
            let key = queue.as_raw() as usize;
            if !s.rb_queue_fences.contains_key(&key) {
                let (Ok(a), Ok(b)) = (s.dev.CreateFence::<ID3D12Fence>(0, D3D12_FENCE_FLAG_NONE), s.dev.CreateFence::<ID3D12Fence>(0, D3D12_FENCE_FLAG_NONE)) else { return };
                s.rb_queue_fences.insert(key, (a, b, 0));
            }
            let (fences, v) = {
                let e = s.rb_queue_fences.get_mut(&key).expect("fences");
                e.2 += 1;
                ((e.0.clone(), e.1.clone()), e.2)
            };
            let _ = queue.Signal(&fences.0, v);
            if !imm.is_empty() {
                s.job_seq += 1;
                s.imm_jobs.push_back(ImmJob { seq: s.job_seq, fence: fences.0.clone(), value: v, writes: imm });
                RB_PENDING.fetch_add(1, Ordering::Release);
                RB_ASYNC.store(true, Ordering::Relaxed);
            }
            if !merged.is_empty() {
                s.readback_async(queue, &merged, &fences, v);
            }
            start_readback_thread();
            return;
        }
        s.rb_value += 1;
        let v = s.rb_value;
        if queue.Signal(&s.rb_fence, v).is_ok() {
            let _ = s.queue.Wait(&s.rb_fence, v);
            super::list_x::note_internal_wait(s.queue.as_raw() as usize, "sparse readback behind a title queue", &s.rb_fence, v);
        }
        s.readback_ranges(&merged);
    }
}

/// Synchronize `[va, va + len)` in a sparse window for a CPU read (`ID3D12Resource::Map` with
/// a non-empty read range, or `Map(0, NULL)` on a sparse readback resource): flushes pending
/// tile clears and unsubmitted CPU-changed chunks first so CPU writes are preserved, then
/// copies the range back from VRAM into CPU memory.
pub unsafe fn readback_now(va: usize, len: usize) {
    if len == 0 {
        return;
    }
    let mut g = lock_state();
    let Some(s) = g.as_mut() else { return };
    if s.window_index(va).is_none() {
        return;
    }
    let zeroed = zero_runs(std::mem::take(&mut s.to_zero));
    if !zeroed.is_empty() {
        for &(a, n) in &zeroed {
            s.clear_sent_range(a, n);
        }
        s.upload_zeros(&zeroed);
    }
    // Preserve any unsubmitted CPU writes on pages overlapping `[va, va + len)`
    let mut dirty_runs: Vec<(usize, usize)> = Vec::new();
    let mut pg = va & !(PAGE - 1);
    let end = va + len;
    while pg < end {
        let t = pg & !(TILE - 1);
        let tile_end = (t + TILE).min(end);
        if let Some(entry) = s.mapped.get_mut(&t) {
            let mut p = pg;
            while p < tile_end {
                let idx = (p - t) / PAGE;
                if entry.holes & (1u16 << idx) == 0 {
                    let bytes = std::slice::from_raw_parts(p as *const u8, PAGE);
                    scan_page(bytes, p, &mut entry.sent[idx], &mut dirty_runs);
                }
                p += PAGE;
            }
        }
        pg = tile_end;
    }
    if !dirty_runs.is_empty() {
        s.upload_dirty(&dirty_runs);
    }
    s.readback_ranges(&[(va, len)]);
}

/// Submits waiting for the state's lock. A thread that takes it again and again – a title mapping
/// its memory a page at a time – can keep them out for seconds: the lock goes to the first to ask
/// once it is free, and the thread that just gave it up is back before a waiting one has woken (a
/// retail title's loading screen stood still for 8 s while another of its threads mapped 1.2 GB).
static SUBMITS_WAITING: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
/// what a commit waits for them at most
const GIVE_WAY_US: u128 = 2000;

/// `m` for a submit: the others let it in first (`lock_in_turn`).
fn lock_first<'a, T>(m: &'a Mutex<T>, waiting: &std::sync::atomic::AtomicUsize) -> std::sync::MutexGuard<'a, T> {
    waiting.fetch_add(1, Ordering::AcqRel);
    let g = m.lock();
    waiting.fetch_sub(1, Ordering::AcqRel);
    g.unwrap()
}

/// `m` once no submit waits for it (or `GIVE_WAY_US` have gone by).
fn lock_in_turn<'a, T>(m: &'a Mutex<T>, waiting: &std::sync::atomic::AtomicUsize) -> std::sync::MutexGuard<'a, T> {
    if waiting.load(Ordering::Acquire) > 0 {
        let t0 = std::time::Instant::now();
        while waiting.load(Ordering::Acquire) > 0 && t0.elapsed().as_micros() < GIVE_WAY_US {
            std::thread::yield_now();
        }
    }
    m.lock().unwrap()
}

/// The title committed `[a, a + n)` in a sparse window. The pages of tiles mapped for it that
/// lie outside the range are not committed (holes); the range's pages are.
pub unsafe fn committed(a: usize, n: usize) {
    if n == 0 {
        return;
    }
    super::title_textures::mark_hot(a, n);
    let mut g = lock_in_turn(&STATE, &SUBMITS_WAITING);
    forget_released(&mut g);
    let Some(s) = g.as_mut() else {
        let (start, end) = page_span(a, n);
        early_add(&mut EARLY.lock().unwrap(), start, end);
        return;
    };
    s.commit_range(a, n);
}

/// The title decommitted `[a, a + n)`: tiles whose pages are all uncommitted (whether in one
/// full-tile decommit or incrementally via sub-tile decommits) give their video memory back
/// (unless a placed buffer still pins the pool tile).
pub unsafe fn decommitted(a: usize, n: usize) {
    if n == 0 {
        return;
    }
    let mut g = lock_in_turn(&STATE, &SUBMITS_WAITING);
    forget_released(&mut g);
    let Some(s) = g.as_mut() else {
        let (start, end) = page_span(a, n);
        early_remove(&mut EARLY.lock().unwrap(), start, end);
        return;
    };
    s.decommit_range(a, n);
}

/// Tiles wait for their mapping call (`State::pending`; read without the lock).
static PENDING: AtomicBool = AtomicBool::new(false);
static PENDING_WAKE: (Mutex<()>, std::sync::Condvar) = (Mutex::new(()), std::sync::Condvar::new());

/// A tile's mapping was put off: sent by the thread started here in 2 ms, unless something sends
/// it before. Work already on the GPU can reach the memory without another submit (it waits for a
/// value the CPU writes).
fn pending_soon() {
    if PENDING.swap(true, Ordering::AcqRel) {
        return;
    }
    static STARTED: std::sync::Once = std::sync::Once::new();
    STARTED.call_once(|| {
        let _ = std::thread::Builder::new().name("xcl-tile-maps".into()).spawn(|| loop {
            {
                let mut g = PENDING_WAKE.0.lock().unwrap_or_else(|e| e.into_inner());
                while !PENDING.load(Ordering::Acquire) {
                    g = PENDING_WAKE.1.wait(g).unwrap_or_else(|e| e.into_inner());
                }
            }
            std::thread::sleep(std::time::Duration::from_millis(2));
            if PENDING.load(Ordering::Acquire) {
                send_pending();
            }
        });
    });
    drop(PENDING_WAKE.0.lock().unwrap_or_else(|e| e.into_inner()));
    PENDING_WAKE.1.notify_one();
}

/// The mappings put off go to the GPU now: the CPU is about to signal a fence, and work submitted
/// before the title committed the memory may run because of it.
pub fn send_pending() {
    // (a signal under the state's lock must not wait for it; the thread sends them then)
    let t0 = std::time::Instant::now();
    loop {
        // (asked each time: another thread may be sending them, and has sent them when the flag goes)
        if !PENDING.load(Ordering::Acquire) {
            return;
        }
        match STATE.try_lock() {
            Ok(mut g) => {
                if let Some(s) = g.as_mut() {
                    unsafe { s.flush_pending() };
                }
                PENDING.store(false, Ordering::Release);
                return;
            }
            Err(std::sync::TryLockError::Poisoned(_)) => return PENDING.store(false, Ordering::Release),
            Err(std::sync::TryLockError::WouldBlock) if t0.elapsed().as_millis() >= 2 => return,
            Err(std::sync::TryLockError::WouldBlock) => std::thread::yield_now(),
        }
    }
}

impl State {
    /// `committed` with the state locked
    unsafe fn commit_range(&mut self, a: usize, n: usize) {
        self.ensure_mapped(a, n, |_| 0xFFFF);
        let mut pg = a & !(PAGE - 1);
        let end = a.saturating_add(n);
        while pg < end {
            let t = pg & !(TILE - 1);
            let tile_end = t.saturating_add(TILE).min(end);
            if let Some(entry) = self.mapped.get_mut(&t) {
                let mut p = pg;
                while p < tile_end {
                    entry.holes &= !(1u16 << ((p - t) / PAGE));
                    p += PAGE;
                }
            }
            if tile_end <= pg {
                break;
            }
            pg = tile_end;
        }
    }

    /// `decommitted` with the state locked: the tiles that go are parked (`flush_parked` unmaps
    /// them and frees their pool tiles)
    unsafe fn decommit_range(&mut self, a: usize, n: usize) {
        let mut pools: Vec<(usize, (u32, u32))> = Vec::new();
        let mut t = a & !(TILE - 1);
        let end = a.saturating_add(n);
        while t < end {
            if let Some(e) = self.mapped.get(&t) {
                pools.push((t, e.pool));
            }
            match t.checked_add(TILE) {
                Some(next) => t = next,
                None => break,
            }
        }
        let tiles = decommit_tiles(&mut self.mapped, &self.pinned, &mut Vec::new(), &mut self.to_zero, a, n);
        if tiles.is_empty() {
            return;
        }
        for &t in &tiles {
            early_remove(&mut self.runs, t, t + TILE);
            if let Ok(i) = pools.binary_search_by_key(&t, |p| p.0) {
                self.parked.insert(t, pools[i].1);
            }
        }
        MAPPED_GEN.fetch_add(1, Ordering::Release);
        if self.parked.len() >= PARKED_MAX && self.flush_parked() {
            self.retire_free_pools();
        }
    }

    /// A pool tile a buffer used at `t` goes back to the free list, unless its window maps it, it
    /// is parked there, or another buffer uses it. Kept mapped for the buffer: parked from here on.
    fn release_if_unused(&mut self, t: usize, tile: (u32, u32)) {
        if self.pinned.contains_key(&tile) {
            return;
        }
        if self.kept.get(&t) == Some(&tile) {
            self.kept.remove(&t);
            self.parked.insert(t, tile);
            return;
        }
        if self.mapped.get(&t).is_none_or(|e| e.pool != tile)
            && self.parked.get(&t) != Some(&tile)
            && !self.quarantined.iter().any(|b| b.tiles.contains(&tile))
        {
            self.free.push(tile);
        }
    }

    unsafe fn note_queue(&mut self, q: &ID3D12CommandQueue) -> bool {
        let key = q.as_raw() as usize;
        if self.queue_fences.contains_key(&key) {
            return true;
        }
        // Ignore queues belonging to a different device (e.g. isolated unit tests).
        let mut qdev: Option<ID3D12Device> = None;
        if q.GetDevice(&mut qdev).is_err() || qdev.as_ref().is_none_or(|d| d.as_raw() != self.dev.as_raw()) {
            return true;
        }
        match self.dev.CreateFence::<ID3D12Fence>(0, D3D12_FENCE_FLAG_NONE) {
            Ok(fence) => {
                self.queue_fences.insert(key, (q.clone(), fence, 0));
                true
            }
            Err(_) => false,
        }
    }

    unsafe fn sync_known_queues(&mut self) -> bool {
        let mut ok = true;
        for raw in super::present::known_queues() {
            if !self.queue_fences.contains_key(&raw) {
                let ptr = raw as *mut c_void;
                if let Some(q) = ID3D12CommandQueue::from_raw_borrowed(&ptr) {
                    ok &= self.note_queue(q);
                }
            }
        }
        ok
    }

    unsafe fn quarantine_tiles(&mut self, tiles: Vec<(u32, u32)>) {
        if tiles.is_empty() {
            return;
        }
        let mut ok = self.sync_known_queues();
        let mut marks = Vec::with_capacity(1 + self.queue_fences.len());
        marks.push((self.fence.clone(), self.value));
        if ok {
            for (q, fence, val) in self.queue_fences.values_mut() {
                let next = *val + 1;
                if q.Signal(&*fence, next).is_ok() {
                    *val = next;
                    marks.push((fence.clone(), next));
                } else {
                    ok = false;
                    break;
                }
            }
        }
        self.quarantined.push_back(RetiredTiles { tiles, marks: ok.then_some(marks) });
    }

    unsafe fn drain_quarantined(&mut self) -> bool {
        let mut freed = false;
        while let Some(front) = self.quarantined.front() {
            let ready = front.marks.as_ref().is_some_and(|marks| marks.iter().all(|(f, v)| f.GetCompletedValue() >= *v));
            if !ready {
                break;
            }
            let batch = self.quarantined.pop_front().unwrap();
            for tile in batch.tiles {
                if !self.pinned.contains_key(&tile) {
                    self.free.push(tile);
                    freed = true;
                }
            }
        }
        freed
    }

    /// The parked tiles unmapped from their windows (one call per window) and their pool tiles
    /// (but those of placed buffers) quarantined behind `self.queue` and every known title
    /// queue's completion fence, then free. Pools left all free are not retired here: callers
    /// do that (`retire_free_pools`) unless they take a tile right away.
    unsafe fn flush_parked(&mut self) -> bool {
        if self.parked.is_empty() {
            return false;
        }
        let mut by_window: FxHashMap<usize, Vec<usize>> = FxHashMap::default();
        let mut tiles: Vec<(u32, u32)> = Vec::new();
        for (t, tile) in std::mem::take(&mut self.parked) {
            // (a placed buffer uses it)
            if self.pinned.contains_key(&tile) {
                self.kept.insert(t, tile);
                continue;
            }
            if let Some(wi) = self.window_index(t) {
                by_window.entry(wi).or_default().push(t);
            }
            tiles.push(tile);
        }
        if by_window.is_empty() {
            // (nothing to unmap: no GPU work reads these through a window)
            self.free.extend(tiles);
            return false;
        }
        for (wi, ts) in by_window {
            let (b, r) = (self.windows[wi].0, self.windows[wi].1.clone());
            self.unmap_tiles(&r, b, &ts);
        }
        self.signal();
        self.quarantine_tiles(tiles);
        true
    }
}

/// The sparse windows' tiles with video memory as sorted (start, end) runs: where a CPU write
/// has to reach the GPU.
pub fn mapped_ranges() -> Vec<(usize, usize)> {
    let g = lock_state();
    let Some(s) = g.as_ref() else { return Vec::new() };
    s.runs.iter().map(|(&a, &b)| (a, b)).collect()
}

/// time spent and pages found by the write-watch scans (`refresh_once`), for the timing line
static WATCH: [AtomicU64; 2] = [AtomicU64::new(0), AtomicU64::new(0)];
/// [lock wait, chunk scan, upload] in µs, runs uploaded
static PHASES: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];

/// Microseconds so far in [looks for CPU writes, waits for the state lock, chunk scans, uploads].
pub fn timing_totals() -> [u64; 4] {
    [WATCH[0].load(Ordering::Relaxed), PHASES[0].load(Ordering::Relaxed), PHASES[1].load(Ordering::Relaxed), PHASES[2].load(Ordering::Relaxed)]
}

pub fn note_watch(t: std::time::Duration, pages: usize) {
    WATCH[0].fetch_add(t.as_micros() as u64, Ordering::Relaxed);
    WATCH[1].fetch_add(pages as u64, Ordering::Relaxed);
}

/// One page to scan: its address and the slot of its hashes (disjoint for every page of one
/// submit: the pages are distinct and their tiles' entries are only touched through these).
struct PageSlot {
    pg: usize,
    slot: *mut Option<Box<[u64; CHUNKS]>>,
}

/// The pages of one submit, shared with the worker threads (`worker_pool`).
struct SharedPages<'a>(&'a [PageSlot]);
// SAFETY: the slots are disjoint, and the scan finishes before the state's lock is given up
unsafe impl Sync for SharedPages<'_> {}

/// Scan `pages` (`scan_page` on each); the changed runs go to `out` in address order.
unsafe fn scan_pages(pages: &[PageSlot], out: &mut Vec<(usize, usize)>) {
    /// pages a worker takes at a time, and the number from which the work is shared
    const PIECE: usize = 64;
    const SHARE_FROM: usize = 256;
    let scan = |p: &PageSlot, out: &mut Vec<(usize, usize)>| {
        let bytes = std::slice::from_raw_parts(p.pg as *const u8, PAGE);
        scan_page(bytes, p.pg, &mut *p.slot, out);
    };
    if pages.len() < SHARE_FROM {
        pages.iter().for_each(|p| scan(p, out));
        return;
    }
    let pieces = pages.len().div_ceil(PIECE);
    let outs: Vec<Mutex<Vec<(usize, usize)>>> = (0..pieces).map(|_| Mutex::new(Vec::new())).collect();
    let shared = SharedPages(pages);
    let work = |k: usize| {
        // (the whole struct, not its field: that is what is Sync)
        let shared = &shared;
        let mut mine = Vec::new();
        for p in &shared.0[k * PIECE..((k + 1) * PIECE).min(shared.0.len())] {
            scan(p, &mut mine);
        }
        *outs[k].lock().unwrap_or_else(|e| e.into_inner()) = mine;
    };
    super::worker_pool::run(pieces, &work);
    for o in outs {
        for (addr, len) in o.into_inner().unwrap_or_else(|e| e.into_inner()) {
            push_run(out, addr, len);
        }
    }
}

/// Copies from the title's memory into the staging ring: shared with the worker threads when
/// there are many bytes.
unsafe fn run_copies(jobs: &mut Vec<(usize, usize, usize)>, staging: *mut u8) {
    const SHARE_FROM: usize = 512 << 10;
    let total: usize = jobs.iter().map(|j| j.2).sum();
    if total < SHARE_FROM || jobs.len() < 2 {
        for &(src, off, n) in jobs.iter() {
            std::ptr::copy_nonoverlapping(src as *const u8, staging.add(off), n);
        }
    } else {
        // pieces of about the same bytes, in the order of the jobs
        let target = (total / 16).max(64 << 10);
        let mut pieces: Vec<(usize, usize)> = Vec::new();
        let (mut from, mut bytes) = (0usize, 0usize);
        for (k, j) in jobs.iter().enumerate() {
            bytes += j.2;
            if bytes >= target {
                pieces.push((from, k + 1));
                from = k + 1;
                bytes = 0;
            }
        }
        if from < jobs.len() {
            pieces.push((from, jobs.len()));
        }
        struct Staging(*mut u8);
        // SAFETY: the copies go to disjoint pieces of the ring, all from the caller's job list
        unsafe impl Sync for Staging {}
        let st = Staging(staging);
        let list: &[(usize, usize, usize)] = jobs;
        super::worker_pool::run(pieces.len(), &|k| {
            let st = &st;
            for &(src, off, n) in &list[pieces[k].0..pieces[k].1] {
                unsafe { std::ptr::copy_nonoverlapping(src as *const u8, st.0.add(off), n) };
            }
        });
    }
    jobs.clear();
}

/// Before the title's ExecuteCommandLists on `queue`: the pages the CPU wrote (`dirty`, 4 KiB
/// pages from the write-watch, sorted) and the reused tiles are copied asynchronously into video
/// memory, and `queue` GPU-waits for that and for new tile mappings.
pub unsafe fn before_execute(queue: &ID3D12CommandQueue, dirty: &[usize], page: usize) {
    let t0 = std::time::Instant::now();
    let mut g = lock_first(&STATE, &SUBMITS_WAITING);
    forget_released(&mut g);
    let Some(s) = g.as_mut() else { return };
    s.drain_readbacks(Drain::Finished);
    // the title's work sees what was committed mapped …
    s.flush_pending();
    PENDING.store(false, Ordering::Release);
    // the title's work sees what was decommitted unmapped, as it was
    s.note_queue(queue);
    let flushed = s.flush_parked();
    let drained = s.drain_quarantined();
    if flushed || drained {
        s.retire_free_pools();
    }
    let t_locked = std::time::Instant::now();
    PHASES[0].fetch_add((t_locked - t0).as_micros() as u64, Ordering::Relaxed);
    let zero_runs = zero_runs(std::mem::take(&mut s.to_zero));
    for &(a, n) in &zero_runs {
        s.clear_sent_range(a, n);
    }
    let mut dirty_runs = std::mem::take(&mut s.scratch_runs);
    dirty_runs.clear();
    let mut pages: Vec<PageSlot> = Vec::with_capacity(dirty.len());
    let mut i = 0;
    while i < dirty.len() {
        let a = dirty[i];
        let mut n = page;
        i += 1;
        while i < dirty.len() && dirty[i] == a + n {
            n += page;
            i += 1;
        }
        // only what has video memory (committed tiles of a sparse window)
        let mut x = a;
        while x < a + n {
            let t = x & !(TILE - 1);
            let end = (t + TILE).min(a + n);
            if let Some(entry) = s.mapped.get_mut(&t) {
                let mut pg = x & !(PAGE - 1);
                while pg < end {
                    let idx = (pg - t) / PAGE;
                    if entry.holes & (1u16 << idx) == 0 {
                        pages.push(PageSlot { pg, slot: &mut entry.sent[idx] });
                    }
                    pg += PAGE;
                }
            }
            x = end;
        }
    }
    scan_pages(&pages, &mut dirty_runs);
    let t_scanned = std::time::Instant::now();
    super::look_stats::scan(pages.len(), &dirty_runs);
    PHASES[1].fetch_add((t_scanned - t_locked).as_micros() as u64, Ordering::Relaxed);
    PHASES[3].fetch_add((zero_runs.len() + dirty_runs.len()) as u64, Ordering::Relaxed);
    // Submit tile clears before CPU writes so overlapping CopyBufferRegion destinations have no WAW hazard.
    if !zero_runs.is_empty() {
        s.upload_zeros(&zero_runs);
    }
    if !dirty_runs.is_empty() {
        s.upload_dirty(&dirty_runs);
    }
    PHASES[2].fetch_add(t_scanned.elapsed().as_micros() as u64, Ordering::Relaxed);
    {
        static CALLS: AtomicU64 = AtomicU64::new(0);
        static BYTES: AtomicU64 = AtomicU64::new(0);
        static MICROS: AtomicU64 = AtomicU64::new(0);
        let bytes: usize = zero_runs.iter().chain(dirty_runs.iter()).map(|r| r.1).sum();
        let n = CALLS.fetch_add(1, Ordering::Relaxed) + 1;
        let b = BYTES.fetch_add(bytes as u64, Ordering::Relaxed) + bytes as u64;
        let us = MICROS.fetch_add(t0.elapsed().as_micros() as u64, Ordering::Relaxed) + t0.elapsed().as_micros() as u64;
        if n.is_power_of_two() && n >= 16 {
            crate::xlog_info!("sparse windows: {n} submits, {} MB of CPU writes uploaded, {} ms spent, {} tiles mapped", b >> 20, us / 1000, s.mapped.len());
            let ms = |x: &AtomicU64| x.load(Ordering::Relaxed) / 1000;
            crate::xlog_info!(
                "sparse windows timing: write-watch {} ms ({} dirty pages), lock wait {} ms, chunk scan {} ms, upload {} ms ({} runs, {} of them in {} dispatches)",
                ms(&WATCH[0]),
                WATCH[1].load(Ordering::Relaxed),
                ms(&PHASES[0]),
                ms(&PHASES[1]),
                ms(&PHASES[2]),
                PHASES[3].load(Ordering::Relaxed),
                SCATTERED[1].load(Ordering::Relaxed),
                SCATTERED[0].load(Ordering::Relaxed)
            );
        }
    }
    dirty_runs.clear();
    s.scratch_runs = dirty_runs;
    let q = queue.as_raw() as usize;
    if s.value > 0 && s.waited.get(&q).copied().unwrap_or(0) < s.value {
        let _ = queue.Wait(&s.fence, s.value);
        super::list_x::note_internal_wait(q, "sparse uploads and tile mappings", &s.fence, s.value);
        s.waited.insert(q, s.value);
    }
}

/// A buffer the title places at `va` in a sparse window (`bytes` long, `desc` the PC
/// description): a reserved resource over the window's tiles there, starting at the 64 KiB
/// below `va` (returned: its distance from there).
pub unsafe fn place_buffer(this: *mut c_void, va: usize, desc: &D3D12_RESOURCE_DESC, state: u32, riid: *const windows_sys::core::GUID, ppv: *mut *mut c_void) -> Option<(crate::util::HRESULT, usize)> {
    let mut g = lock_state();
    let s = g.as_mut()?;
    let start = va & !(TILE - 1);
    let delta = va - start;
    let bytes = desc.Width as usize;
    if bytes == 0 {
        return None;
    }
    s.window_index(start)?;
    s.window_index(va.checked_add(bytes)?.checked_sub(1)?)?;
    let mut d = *desc;
    d.Width = (delta + bytes) as u64;
    d.Alignment = 0;
    d.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    // Reserved buffers only accept NONE or ALLOW_UNORDERED_ACCESS (strip console flags and ALLOW_CROSS_ADAPTER).
    d.Flags &= D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    if state & (D3D12_RESOURCE_STATE_UNORDERED_ACCESS.0 as u32 | D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE.0 as u32) != 0 {
        d.Flags |= D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    }
    let hr = super::real::dev_create_reserved_resource(this, &d as *const _ as usize, state as usize, 0, riid, ppv);
    if hr < 0 || ppv.is_null() || (*ppv).is_null() {
        return None;
    }
    // its tiles: the window's video memory at the same addresses; probe actual OS commitment
    // so uncommitted pages (e.g. [start, va) or reserved tails) stay marked as holes
    s.ensure_mapped(start, delta + bytes, |t| query_tile_holes(t));
    let tiles: Vec<(usize, (u32, u32))> = (start..va + bytes).step_by(TILE).filter_map(|t| s.mapped.get(&t).map(|e| (t, e.pool))).collect();
    let raw = *ppv;
    let res = ID3D12Resource::from_raw_borrowed(&raw)?.clone();
    s.map_tiles(&res, start, &tiles);
    for &(_, tile) in &tiles {
        *s.pinned.entry(tile).or_default() += 1;
    }
    s.buffers.insert(raw as usize, tiles);
    s.signal();
    Some((hr, delta))
}

/// Placed buffers released while `STATE` was held, let go by the next `lock_state`. Held only
/// to push or take, never across a call out.
static FORGOTTEN: Mutex<Vec<usize>> = Mutex::new(Vec::new());

/// `STATE`, with the buffers released while it was held let go first.
fn lock_state() -> std::sync::MutexGuard<'static, Option<State>> {
    let mut g = STATE.lock().unwrap();
    forget_released(&mut g);
    g
}

fn forget_released(g: &mut Option<State>) {
    let gone = std::mem::take(&mut *FORGOTTEN.lock().unwrap());
    let Some(s) = g.as_mut() else { return };
    for res in gone {
        forget_in(s, res);
    }
}

/// A placed buffer is gone: its tiles are not pinned by it any more. This runs from the
/// resource's last Release, which D3D12 can make while it holds a lock of its own (in a title's
/// command list Reset), and `before_execute` calls D3D12 with `STATE` held: waiting for `STATE`
/// here deadlocked the two threads. So it does not wait: with `STATE` held, the holder's next
/// `lock_state` lets the buffer go.
pub fn forget_buffer(res: usize) {
    FORGOTTEN.lock().unwrap().push(res);
    match STATE.try_lock() {
        Ok(mut g) => forget_released(&mut g),
        Err(std::sync::TryLockError::WouldBlock) => {}
        Err(e) => panic!("sparse state: {e}"),
    }
}

fn forget_in(s: &mut State, res: usize) {
    let Some(tiles) = s.buffers.remove(&res) else { return };
    for (t, tile) in tiles {
        if let Some(c) = s.pinned.get_mut(&tile) {
            *c -= 1;
            if *c == 0 {
                s.pinned.remove(&tile);
                s.release_if_unused(t, tile);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The runs of the tiles mapped, made from all of them.
    fn runs_of(s: &State) -> BTreeMap<usize, usize> {
        let mut tiles: Vec<usize> = s.mapped.keys().copied().collect();
        tiles.sort_unstable();
        let mut runs: Vec<(usize, usize)> = Vec::new();
        for t in tiles {
            match runs.last_mut() {
                Some(last) if last.1 == t => last.1 = t + TILE,
                _ => runs.push((t, t + TILE)),
            }
        }
        runs.into_iter().collect()
    }

    /// A device of the test's own (a software one without a GPU).
    unsafe fn test_device() -> ID3D12Device {
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
        use windows::Win32::Graphics::Dxgi::*;
        let factory: IDXGIFactory4 = CreateDXGIFactory1().expect("DXGI factory");
        let mut dev: Option<ID3D12Device> = None;
        if std::env::var_os("XCL_TEST_WARP").is_some() || D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
            let warp: IDXGIAdapter = factory.EnumWarpAdapter().expect("WARP");
            D3D12CreateDevice(&warp, D3D_FEATURE_LEVEL_11_0, &mut dev).expect("WARP D3D12 device");
        }
        dev.expect("D3D12 device")
    }

    /// Whether `dev` is a software device (WARP).
    unsafe fn software(dev: &ID3D12Device) -> bool {
        use windows::Win32::Graphics::Dxgi::*;
        let Ok(factory) = CreateDXGIFactory1::<IDXGIFactory4>() else { return false };
        let adapter: windows::core::Result<IDXGIAdapter1> = factory.EnumAdapterByLuid(dev.GetAdapterLuid());
        adapter.and_then(|a| a.GetDesc1()).is_ok_and(|d| d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 != 0)
    }

    /// The small ones of an upload's runs go up in dispatches: each lands where it belongs (far
    /// into the window too, next to runs that go up with a copy), and what lies between them keeps
    /// what the GPU has there.
    #[test]
    fn small_runs_go_up_in_dispatches_and_leave_what_lies_between() {
        use windows_sys::Win32::System::Memory::{MEM_RELEASE, MEM_RESERVE};
        const SIZE: usize = 3 << 30;
        unsafe {
            let dev = test_device();
            // (the software device makes a reserved buffer of 3 GiB and is removed by the first copies
            // into it, 0x887a0020 – and a process has one device an adapter: every test after it failed)
            if software(&dev) {
                return crate::gpu::recomp::runner::skip_test("no GPU");
            }
            let mut reserved: Option<ID3D12Resource> = None;
            if dev.CreateReservedResource(&buffer_desc(SIZE as u64), D3D12_RESOURCE_STATE_COMMON, None, &mut reserved).is_err() {
                return crate::gpu::recomp::runner::skip_test("no GPU");
            }
            let window = reserved.expect("reserved buffer");
            let base = VirtualAlloc(std::ptr::null(), SIZE, MEM_RESERVE, PAGE_NOACCESS) as usize;
            assert!(base != 0 && base % TILE == 0);
            let mut s = new_state(&dev, vec![(base, window.clone())]).expect("state");
            // (the 1 GiB a dispatch reaches from its base ends between the third and the fourth)
            let tiles = [base, base + TILE, base + (1 << 30) - TILE, base + (1 << 30), base + SIZE - TILE];
            let mut seed = 0x1234_5678_9ABC_DEF1u64;
            let mut random = || {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                seed
            };
            for &t in &tiles {
                assert!(!VirtualAlloc(t as *const c_void, TILE, MEM_COMMIT, PAGE_READWRITE).is_null());
                s.commit_range(t, TILE);
                for i in 0..TILE / 8 {
                    *(t as *mut u64).add(i) = random();
                }
            }
            // all of it goes up, then the CPU writes all of it again and only some runs go up
            let all: Vec<(usize, usize)> = tiles.iter().map(|&t| (t, TILE)).collect();
            s.upload_dirty(&all);
            let on_gpu: Vec<Vec<u8>> = tiles.iter().map(|&t| std::slice::from_raw_parts(t as *const u8, TILE).to_vec()).collect();
            let mut runs: Vec<(usize, usize)> = Vec::new();
            for &t in &tiles {
                for i in 0..TILE / 8 {
                    *(t as *mut u64).add(i) = random();
                }
                let mut o = 0;
                loop {
                    o += CHUNK * (1 + random() as usize % 9);
                    let n = if random() % 23 == 0 { SCATTER_RUN + CHUNK * (random() as usize % 40) } else { CHUNK * (1 + random() as usize % 12) };
                    if o + n > TILE {
                        break;
                    }
                    runs.push((t + o, n));
                    o += n;
                }
            }
            let small = runs.iter().filter(|r| r.1 < SCATTER_RUN).count() as u64;
            let counts = || (SCATTERED[0].load(Ordering::Relaxed), SCATTERED[1].load(Ordering::Relaxed));
            let before = counts();
            s.upload_dirty(&runs);
            let after = counts();
            assert!(small > 300 && runs.len() as u64 > small, "{small} small runs of {}", runs.len());
            assert!(after.0 - before.0 >= 3 && after.1 - before.1 >= small, "{} dispatches of {} runs", after.0 - before.0, after.1 - before.1);

            let mut d = buffer_desc((tiles.len() * TILE) as u64);
            d.Flags = D3D12_RESOURCE_FLAG_NONE;
            let mut back: Option<ID3D12Resource> = None;
            dev.CreateCommittedResource(&D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() }, D3D12_HEAP_FLAG_NONE, &d, D3D12_RESOURCE_STATE_COPY_DEST, None, &mut back).expect("readback buffer");
            let back = back.expect("readback buffer");
            let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).expect("alloc");
            let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None).expect("list");
            for (k, &t) in tiles.iter().enumerate() {
                list.CopyBufferRegion(&back, (k * TILE) as u64, &window, (t - base) as u64, TILE as u64);
            }
            list.Close().expect("Close");
            s.queue.ExecuteCommandLists(&[Some(list.cast().expect("list"))]);
            s.signal();
            s.wait_for(s.value);
            let mut p = std::ptr::null_mut();
            back.Map(0, None, Some(&mut p)).expect("Map");
            for (k, &t) in tiles.iter().enumerate() {
                let mut want = on_gpu[k].clone();
                for &(a, n) in runs.iter().filter(|r| r.0 >= t && r.0 < t + TILE) {
                    want[a - t..a - t + n].copy_from_slice(std::slice::from_raw_parts(a as *const u8, n));
                }
                let got = std::slice::from_raw_parts((p as *const u8).add(k * TILE), TILE);
                let wrong = (0..TILE / CHUNK).filter(|c| got[c * CHUNK..][..CHUNK] != want[c * CHUNK..][..CHUNK]).collect::<Vec<_>>();
                assert!(wrong.is_empty(), "tile {k}: chunks {:?}… of {} are not what they should be", &wrong[..wrong.len().min(8)], TILE / CHUNK);
            }
            drop(s);
            assert_ne!(VirtualFree(base as *mut c_void, 0, MEM_RELEASE), 0);
        }
    }

    /// A copy back after a title submission does not wait for the title's queue: that queue can
    /// be waiting for work the title submits next, from the thread that is in here. The copy
    /// lands once the title's work is done (`drain_readbacks`). Before, the sparse queue
    /// GPU-waited for the title's queue and this thread waited for the copy (5 s, then on).
    #[test]
    fn a_copy_back_does_not_wait_for_the_title_queue() {
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
        use windows::Win32::Graphics::Dxgi::*;
        use windows_sys::Win32::System::Memory::{MEM_RELEASE, MEM_RESERVE};
        unsafe {
            let factory: IDXGIFactory4 = CreateDXGIFactory1().expect("DXGI factory");
            let mut dev: Option<ID3D12Device> = None;
            if D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
                let warp: IDXGIAdapter = factory.EnumWarpAdapter().expect("WARP");
                D3D12CreateDevice(&warp, D3D_FEATURE_LEVEL_11_0, &mut dev).expect("WARP D3D12 device");
            }
            let dev = dev.expect("D3D12 device");
            let base = VirtualAlloc(std::ptr::null(), TILE, MEM_RESERVE, PAGE_NOACCESS) as usize;
            assert!(base != 0 && base % TILE == 0);
            assert!(!VirtualAlloc(base as *const c_void, TILE, MEM_COMMIT, PAGE_READWRITE).is_null());
            let mut reserved: Option<ID3D12Resource> = None;
            dev.CreateReservedResource(&buffer_desc(TILE as u64), D3D12_RESOURCE_STATE_COMMON, None, &mut reserved).expect("CreateReservedResource");
            let reserved = reserved.expect("reserved buffer");
            let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).expect("queue");
            let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).expect("fence");
            let rb_fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).expect("rb_fence");
            let event = windows::Win32::System::Threading::CreateEventW(None, false, false, None).expect("event");
            let rb_alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).expect("rb_alloc");
            let rb_list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &rb_alloc, None).expect("rb_list");
            rb_list.Close().expect("rb_list close");
            let rbq_queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).expect("readback queue");
            let rbq_fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).expect("readback fence");

            let mut s = State {
                dev: dev.clone(),
                queue,
                fence,
                value: 0,
                event,
                rb_fence,
                rb_value: 0,
                rb_queue_fences: FxHashMap::default(),
                rbq: Some((rbq_queue, rbq_fence)),
                rbq_value: 0,
                rb_jobs: VecDeque::new(),
                rb_pool: Vec::new(),
                imm_jobs: VecDeque::new(),
                job_seq: 0,
                windows: vec![(base, reserved.clone())],
                pools: Vec::new(),
                free: Vec::new(),
                retired: Vec::new(),
                mapped: FxHashMap::default(),
                pinned: FxHashMap::default(),
                parked: FxHashMap::default(),
                quarantined: VecDeque::new(),
                queue_fences: FxHashMap::default(),
                kept: FxHashMap::default(),
                pending: Vec::new(),
                buffers: FxHashMap::default(),
                to_zero: BTreeSet::new(),
                staging: None,
                ring_write: 0,
                ring_submitted: 0,
                ring_free: 0,
                inflight: VecDeque::new(),
                zero_buf: None,
                slots: Vec::new(),
                next_slot: 0,
                readback: None,
                rb_alloc,
                rb_list,
                rb_last_fence: 0,
                waited: FxHashMap::default(),
                runs: BTreeMap::new(),
                scratch_runs: Vec::new(),
            };
            s.commit_range(base, TILE);
            assert!(s.mapped.contains_key(&base));
            // (as before the title's submit)
            s.flush_pending();
            std::ptr::write_bytes(base as *mut u8, 0, 256);

            // the title's queue: blocked on a fence nothing has signalled yet, then a copy of
            // 0..64 into the tile (a GPU write the title's next CPU read expects)
            let title: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).expect("title queue");
            let gate: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).expect("gate");
            let mut up: Option<ID3D12Resource> = None;
            dev.CreateCommittedResource(&D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_UPLOAD, ..Default::default() }, D3D12_HEAP_FLAG_NONE, &{ let mut d = buffer_desc(256); d.Flags = D3D12_RESOURCE_FLAG_NONE; d }, D3D12_RESOURCE_STATE_GENERIC_READ, None, &mut up)
                .expect("upload buffer");
            let up = up.expect("upload buffer");
            let mut p: *mut c_void = std::ptr::null_mut();
            up.Map(0, None, Some(&mut p)).expect("Map");
            for i in 0..64u32 {
                *(p as *mut u32).add(i as usize) = 0xC0DE_0000 | i;
            }
            let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).expect("alloc");
            let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None).expect("list");
            list.CopyBufferRegion(&reserved, 0, &up, 0, 256);
            list.Close().expect("close");
            title.Wait(&s.fence, s.value).expect("wait for the tile mapping");
            title.Wait(&gate, 1).expect("wait for the gate");
            title.ExecuteCommandLists(&[Some(list.cast::<ID3D12CommandList>().unwrap())]);

            let t0 = std::time::Instant::now();
            s.after_execute(&title, &[(base, 256)], &[]);
            let took = t0.elapsed();
            assert!(took < std::time::Duration::from_secs(1), "the copy back waited {took:?} for the title's queue");
            s.drain_readbacks(Drain::Finished);
            assert_eq!(*(base as *const u32), 0, "nothing lands before the title's work is done");

            // the title's queue goes on: the copy back lands once it is done
            gate.Signal(1).expect("gate signal");
            // as the title would: wait for its own work (here the fences its queue signals: the
            // first once its lists are done, the second once the copy behind them is)
            let t = std::time::Instant::now();
            loop {
                s.drain_readbacks(Drain::TitleDone);
                if *(base as *const u32) == 0xC0DE_0000 {
                    break;
                }
                assert!(t.elapsed() < std::time::Duration::from_secs(5), "the copy back did not land");
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
            for i in 0..64u32 {
                assert_eq!(*(base as *const u32).add(i as usize), 0xC0DE_0000 | i, "dword {i} after the copy back");
            }
            assert!(s.rb_jobs.is_empty(), "the copy back was consumed");
            assert_eq!(s.rb_pool.len(), 1, "its buffer is kept for the next one");
            drop(s);
            VirtualFree(base as *mut c_void, 0, MEM_RELEASE);
        }
    }

    /// `queue()` answers while the state's lock is held: a resource's last `Release` asks for
    /// it (`late_release::drain_known`) from inside a D3D12 call that the lock's holder can be
    /// waiting for (a list's `Reset` against an upload's allocator `Reset`, with `XCL_DRED=1`).
    #[test]
    fn the_queue_is_answered_while_the_state_is_locked() {
        let held = STATE.lock().unwrap();
        let (tx, rx) = std::sync::mpsc::channel();
        let asked = std::thread::spawn(move || {
            let _ = queue();
            let _ = tx.send(());
        });
        let answered = rx.recv_timeout(std::time::Duration::from_secs(5)).is_ok();
        drop(held);
        asked.join().unwrap();
        assert!(answered, "queue() waited for the sparse state's lock");
    }

    /// A submit gets the lock soon although another thread takes it again and again for 0.2 ms each.
    #[test]
    fn a_submit_is_let_in_before_the_commits() {
        use std::sync::atomic::AtomicUsize;
        let (m, waiting, stop) = (Mutex::new(0u64), AtomicUsize::new(0), AtomicBool::new(false));
        std::thread::scope(|s| {
            s.spawn(|| {
                while !stop.load(Ordering::Acquire) {
                    let mut g = lock_in_turn(&m, &waiting);
                    *g += 1;
                    let t = std::time::Instant::now();
                    while t.elapsed().as_micros() < 200 {
                        std::hint::spin_loop();
                    }
                }
            });
            while *m.lock().unwrap() == 0 {
                std::thread::yield_now();
            }
            let mut turns: Vec<u64> = Vec::new();
            for _ in 0..20 {
                let before = *lock_first(&m, &waiting);
                std::thread::sleep(std::time::Duration::from_millis(2));
                let after = *lock_first(&m, &waiting);
                // (the turns the other thread had while this one slept and waited)
                turns.push(after - before);
            }
            stop.store(true, Ordering::Release);
            turns.sort_unstable();
            // 2 ms of sleep are 10 turns; without the rule a waiting thread stood behind hundreds
            assert!(turns[10] < 60, "the other thread's turns before a submit had the lock: {turns:?}");
        });
        assert_eq!(waiting.load(Ordering::Acquire), 0);
    }

    /// A mapping put off does not wait for the title's next submit: a CPU signal sends it, and
    /// without one it goes a few milliseconds after the commit (work the title submitted before
    /// may reach the memory once a value the CPU writes lets it run).
    #[test]
    fn a_mapping_put_off_is_sent_without_a_submit() {
        if super::super::backend() != super::super::Backend::Real {
            return;
        }
        unsafe {
            let _ = super::super::real::device_raw();
            // (a device without sparse windows: commits wait for one, nothing is put off)
            if STATE.lock().unwrap().is_none() {
                return crate::gpu::recomp::runner::skip_test("no GPU");
            }
            let Some((at, len)) = alloc(64 * TILE, TILE) else { return };
            let put_off = || STATE.lock().unwrap().as_ref().is_some_and(|s| s.pending.iter().any(|t| (at..at + len).contains(t)));
            let mapped = || STATE.lock().unwrap().as_ref().is_some_and(|s| (at..at + len).step_by(TILE).all(|t| s.mapped.contains_key(&t)));
            send_pending();
            assert!(mapped() && !put_off(), "a CPU signal sends what is put off");
            decommitted(at, len);
            VirtualFree(at as *mut c_void, len, MEM_DECOMMIT);
            // (as a submit does: the tiles unmapped, so the next commit needs mappings again)
            if let Some(s) = STATE.lock().unwrap().as_mut() {
                s.flush_parked();
            }
            assert!(!VirtualAlloc(at as *const c_void, len, MEM_COMMIT, PAGE_READWRITE).is_null());
            committed(at, len);
            let t0 = std::time::Instant::now();
            while put_off() {
                assert!(t0.elapsed().as_secs() < 5, "nothing sent the mappings");
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
            assert!(mapped());
            assert!(free(at));
        }
    }

    /// A block freed on one thread while another allocates: its address goes back to the
    /// windows only once it is decommitted (released first, the decommit came after another
    /// thread had the same address again: that thread's writes faulted).
    #[test]
    fn a_freed_block_is_decommitted_before_its_address_is_reused() {
        if crate::gpu::uma::get().is_none_or(|u| !u.has_sparse()) {
            return;
        }
        let threads: Vec<_> = (0..4)
            .map(|_| {
                std::thread::spawn(|| unsafe {
                    for i in 0..2000u64 {
                        let Some((p, n)) = alloc(TILE, TILE) else { continue };
                        std::ptr::write_volatile(p as *mut u64, i);
                        std::ptr::write_volatile((p + n - 8) as *mut u64, i);
                        assert_eq!(std::ptr::read_volatile(p as *const u64), i);
                        assert!(free(p));
                    }
                })
            })
            .collect();
        for t in threads {
            t.join().unwrap();
        }
    }

    /// `f` on another thread while this one holds the sparse state (as `before_execute` does
    /// around its D3D12 calls, which can wait for that thread): did `f` finish without waiting
    /// for the state? The thread runs before the state is taken, so its start is not timed.
    fn returns_while_the_state_is_held(f: impl FnOnce() + Send + 'static) -> bool {
        let (ready_tx, ready_rx) = std::sync::mpsc::channel();
        let (go_tx, go_rx) = std::sync::mpsc::channel::<()>();
        let (tx, rx) = std::sync::mpsc::channel();
        let t = std::thread::spawn(move || {
            let _ = ready_tx.send(());
            let _ = go_rx.recv();
            f();
            let _ = tx.send(());
        });
        ready_rx.recv().unwrap();
        let held = STATE.lock().unwrap();
        go_tx.send(()).unwrap();
        let done = rx.recv_timeout(std::time::Duration::from_secs(5));
        drop(held);
        let joined = t.join();
        assert!(joined.is_ok(), "the thread panicked: {joined:?}");
        done.is_ok()
    }

    /// A placed buffer released while another thread holds the sparse state: the release does
    /// not wait for the state, and the next `lock_state` lets the buffer go.
    #[test]
    fn a_buffer_released_while_the_state_is_held_does_not_wait_for_it() {
        const RES: usize = 0x7ff0_dead_0000;
        assert!(returns_while_the_state_is_held(|| forget_buffer(RES)), "forget_buffer waited for the sparse state's lock");
        drop(lock_state());
        assert!(!FORGOTTEN.lock().unwrap().contains(&RES));
    }


    /// Freeing the last tile in use of a pool and taking a tile again takes the retired pool
    /// back, not a new 64 MB heap each time (#913: 288 pools, 18 GB asked of a 12 GB card).
    #[test]
    fn a_pool_retired_a_moment_ago_is_taken_back() {
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
        use windows::Win32::Graphics::Dxgi::*;

        unsafe {
            let mut dev: Option<ID3D12Device> = None;
            if D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
                let factory: IDXGIFactory4 = CreateDXGIFactory1().expect("DXGI factory");
                let warp: IDXGIAdapter = factory.EnumWarpAdapter().expect("WARP");
                D3D12CreateDevice(&warp, D3D_FEATURE_LEVEL_11_0, &mut dev).expect("WARP D3D12 device");
            }
            let dev = dev.expect("D3D12 device");
            let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).expect("queue");
            let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).expect("fence");
            let rb_fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).expect("rb_fence");
            let event = windows::Win32::System::Threading::CreateEventW(None, false, false, None).expect("event");
            let rb_alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).expect("rb_alloc");
            let rb_list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &rb_alloc, None).expect("rb_list");
            rb_list.Close().expect("rb_list close");
            let mut s = State {
                dev,
                queue,
                fence,
                value: 0,
                event,
                rb_fence,
                rb_value: 0,
                rb_queue_fences: FxHashMap::default(),
                rbq: None,
                rbq_value: 0,
                rb_jobs: VecDeque::new(),
                rb_pool: Vec::new(),
                imm_jobs: VecDeque::new(),
                job_seq: 0,
                windows: Vec::new(),
                pools: Vec::new(),
                free: Vec::new(),
                retired: Vec::new(),
                parked: FxHashMap::default(),
                quarantined: VecDeque::new(),
                queue_fences: FxHashMap::default(),
                kept: FxHashMap::default(),
                pending: Vec::new(),
                mapped: FxHashMap::default(),
                pinned: FxHashMap::default(),
                buffers: FxHashMap::default(),
                to_zero: BTreeSet::new(),
                staging: None,
                ring_write: 0,
                ring_submitted: 0,
                ring_free: 0,
                inflight: VecDeque::new(),
                zero_buf: None,
                slots: Vec::new(),
                next_slot: 0,
                readback: None,
                rb_alloc,
                rb_list,
                rb_last_fence: 0,
                waited: FxHashMap::default(),
                runs: BTreeMap::new(),
                scratch_runs: Vec::new(),
            };

            let (first, reused) = s.take_tile().expect("a tile from a new pool");
            assert!(!reused, "a new pool's tile needs no clearing");
            s.free.push(first);
            s.retire_free_pools();
            assert_eq!(s.retired.len(), 1, "an all-free pool is retired");
            for round in 0..8 {
                let (tile, reused) = s.take_tile().expect("a tile");
                assert!(reused, "round {round}: a tile of a retired pool is cleared before use");
                s.free.push(tile);
                s.retire_free_pools();
                assert_eq!(s.retired.len(), 1, "round {round}: the retired pool was taken back, not a new one made");
                assert_eq!(s.pools.iter().filter(|p| p.is_some()).count(), 0, "round {round}");
            }
        }
    }

    #[test]
    fn aligned_tile_addresses_spread_across_hash_buckets() {
        use std::hash::BuildHasher;

        let hasher = BuildHasherDefault::<FxHasher>::default();
        let mut used = [false; 256];
        for tile in 0..used.len() {
            let address = (1usize << 32) + tile * TILE;
            let bucket = hasher.hash_one(address) as usize & (used.len() - 1);
            used[bucket] = true;
        }
        let occupied = used.iter().filter(|&&bucket| bucket).count();
        assert!(occupied >= used.len() / 2, "256 aligned addresses used {occupied} buckets");
    }

    /// Only the 64-byte chunks the CPU changed are sent again – the rest of the page may hold
    /// what the GPU wrote.
    #[test]
    fn only_changed_chunks_are_sent() {
        let mut page = vec![0u8; PAGE];
        let mut sent = ZERO_HASHES;
        assert!(changed_chunks(&page, &mut sent).is_empty(), "a zero page is what the GPU has");
        page[10] = 1;
        page[200] = 2;
        page[255] = 3;
        page[4000] = 4;
        assert_eq!(changed_chunks(&page, &mut sent), vec![(0, 64), (192, 64), (3968, 64)]);
        assert!(changed_chunks(&page, &mut sent).is_empty(), "nothing changed since");
        page[256] = 5; // next to the chunk at 192: one run
        page[200] = 9;
        assert_eq!(changed_chunks(&page, &mut sent), vec![(192, 128)]);
    }

    /// `scan_page` skips allocating a hash box on all-zero pages, allocates on the first write,
    /// and coalesces adjacent dirty chunks across consecutive pages.
    #[test]
    fn scan_page_lazy_alloc_and_coalesce() {
        let mut page = vec![0u8; PAGE];
        let mut slot: Option<Box<[u64; CHUNKS]>> = None;
        let mut out = Vec::new();
        scan_page(&page, 0x1_0000, &mut slot, &mut out);
        assert!(slot.is_none(), "all-zero page must not allocate hash box");
        assert!(out.is_empty());

        page[PAGE - 64] = 0xAA;
        scan_page(&page, 0x1_0000, &mut slot, &mut out);
        assert!(slot.is_some());
        assert_eq!(out, vec![(0x1_0FC0, 64)]);

        // Adjacent chunk at the start of the next page coalesces into a single run.
        let mut next = vec![0u8; PAGE];
        next[0] = 0xBB;
        let mut next_slot: Option<Box<[u64; CHUNKS]>> = None;
        scan_page(&next, 0x1_1000, &mut next_slot, &mut out);
        assert_eq!(out, vec![(0x1_0FC0, 128)]);
    }

    /// Ring-buffer planner keeps every chunk contiguous, 64-byte aligned, <= cap/2, and splits
    /// cleanly across the end of the ring buffer without wrapping mid-copy.
    #[test]
    fn ring_planner_handles_wrap_and_large_chunks() {
        let cap = 64 << 20;
        // Normal aligned chunk
        let (a, n) = plan_ring_chunk(0, 4096, cap);
        assert_eq!((a, n), (0, 4096));
        // Unaligned cursor aligns up to 64
        let (a, n) = plan_ring_chunk(17, 256, cap);
        assert_eq!((a, n), (64, 256));
        // Oversized chunk capped at cap / 2
        let (a, n) = plan_ring_chunk(0, cap * 2, cap);
        assert_eq!((a, n), (0, cap / 2));
        // Near end of ring with >= 4 KiB left: shrinks to fill tail
        let near = (cap - 8192) as u64;
        let (a, n) = plan_ring_chunk(near, 16384, cap);
        assert_eq!((a, n), (near, 8192));
        assert_eq!((a as usize % cap) + n, cap);
        // Tiny tail (< 4 KiB): wraps cleanly to 0 of next cycle
        let tiny = (cap - 512) as u64;
        let (a, n) = plan_ring_chunk(tiny, 4096, cap);
        assert_eq!((a, n), (cap as u64, 4096));
        assert_eq!(a as usize % cap, 0);
    }

    /// `query_tile_holes` marks reserved, free, and `PAGE_NOACCESS` pages as holes while clearing
    /// bits only for committed, accessible 4 KiB pages.
    #[test]
    fn query_tile_holes_detects_reserved_noaccess_and_committed_pages() {
        use windows_sys::Win32::System::Memory::{MEM_RELEASE, MEM_RESERVE};

        unsafe {
            // SAFETY: Reserving 2 tiles (128 KiB) with VirtualAlloc returns a 64 KiB-aligned VA.
            let base = VirtualAlloc(std::ptr::null(), 2 * TILE, MEM_RESERVE, PAGE_NOACCESS) as usize;
            assert!(base != 0 && base % TILE == 0);

            // Initially every page in both reserved tiles is a hole.
            assert_eq!(query_tile_holes(base), 0xFFFF);
            assert_eq!(query_tile_holes(base + TILE), 0xFFFF);

            // Commit pages 4..8 as PAGE_READWRITE and page 9 as PAGE_NOACCESS.
            let p4 = VirtualAlloc((base + 4 * PAGE) as *const c_void, 4 * PAGE, MEM_COMMIT, PAGE_READWRITE);
            assert!(!p4.is_null());
            let p9 = VirtualAlloc((base + 9 * PAGE) as *const c_void, PAGE, MEM_COMMIT, PAGE_NOACCESS);
            assert!(!p9.is_null());

            // Only bits 4..8 are cleared; reserved [0..4), PAGE_NOACCESS [9], and reserved [10..16) remain holes.
            assert_eq!(query_tile_holes(base), 0xFFFF & !(0b1111u16 << 4));
            assert_eq!(query_tile_holes(base + TILE), 0xFFFF);

            // Commit pages 0..4 and confirm bits 0..4 clear too.
            let p0 = VirtualAlloc(base as *const c_void, 4 * PAGE, MEM_COMMIT, PAGE_READWRITE);
            assert!(!p0.is_null());
            assert_eq!(query_tile_holes(base), 0xFF00);

            assert_ne!(VirtualFree(base as *mut c_void, 0, MEM_RELEASE), 0);
            // Released (MEM_FREE) memory reports all holes.
            assert_eq!(query_tile_holes(base), 0xFFFF);
        }
    }

    /// Mapping tiles for a placed buffer over partially-committed/reserved memory must preserve
    /// uncommitted hole bits so `readback_ranges` / `sync_sent_range` do not access-violate,
    /// and later commits clear those hole bits so subsequent writes scan cleanly.
    #[test]
    fn ensure_mapped_with_query_holes_skips_uncommitted_pages_on_readback() {
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
        use windows::Win32::Graphics::Dxgi::*;
        use windows_sys::Win32::System::Memory::{MEM_RELEASE, MEM_RESERVE};

        unsafe {
            let factory: IDXGIFactory4 = CreateDXGIFactory1().expect("DXGI factory");
            let mut dev: Option<ID3D12Device> = None;
            if D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
                let warp: IDXGIAdapter = factory.EnumWarpAdapter().expect("WARP");
                D3D12CreateDevice(&warp, D3D_FEATURE_LEVEL_11_0, &mut dev).expect("WARP D3D12 device");
            }
            let dev = dev.expect("D3D12 device");

            // Reserve 2 tiles; commit only [base + 0x4000, base + 0x8000) to simulate a
            // non-64KiB-aligned placed buffer with uncommitted prefix [base, base + 0x4000) and
            // uncommitted second tile [base + TILE, base + 2*TILE).
            let base = VirtualAlloc(std::ptr::null(), 2 * TILE, MEM_RESERVE, PAGE_NOACCESS) as usize;
            assert!(base != 0 && base % TILE == 0);
            assert!(!VirtualAlloc((base + 4 * PAGE) as *const c_void, 4 * PAGE, MEM_COMMIT, PAGE_READWRITE).is_null());

            let mut reserved: Option<ID3D12Resource> = None;
            dev.CreateReservedResource(&buffer_desc((2 * TILE) as u64), D3D12_RESOURCE_STATE_COMMON, None, &mut reserved)
                .expect("CreateReservedResource");
            let reserved = reserved.expect("reserved buffer");
            let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).expect("queue");
            let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).expect("fence");
            let rb_fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).expect("rb_fence");
            let event = windows::Win32::System::Threading::CreateEventW(None, false, false, None).expect("event");
            let rb_alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).expect("rb_alloc");
            let rb_list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &rb_alloc, None).expect("rb_list");
            rb_list.Close().expect("rb_list close");

            let mut s = State {
                dev,
                queue,
                fence,
                value: 0,
                event,
                rb_fence,
                rb_value: 0,
                rb_queue_fences: FxHashMap::default(),
                rbq: None,
                rbq_value: 0,
                rb_jobs: VecDeque::new(),
                rb_pool: Vec::new(),
                imm_jobs: VecDeque::new(),
                job_seq: 0,
                windows: vec![(base, reserved)],
                pools: Vec::new(),
                free: Vec::new(),
                retired: Vec::new(),
                mapped: FxHashMap::default(),
                pinned: FxHashMap::default(),
                parked: FxHashMap::default(),
                quarantined: VecDeque::new(),
                queue_fences: FxHashMap::default(),
                kept: FxHashMap::default(),
                pending: Vec::new(),
                buffers: FxHashMap::default(),
                to_zero: BTreeSet::new(),
                staging: None,
                ring_write: 0,
                ring_submitted: 0,
                ring_free: 0,
                inflight: VecDeque::new(),
                zero_buf: None,
                slots: Vec::new(),
                next_slot: 0,
                readback: None,
                rb_alloc,
                rb_list,
                rb_last_fence: 0,
                waited: FxHashMap::default(),
                runs: BTreeMap::new(),
                scratch_runs: Vec::new(),
            };

            // Mirrors `place_buffer`: map both tiles probing real OS page commitment.
            s.ensure_mapped(base, 2 * TILE, |t| query_tile_holes(t));
            assert_eq!(s.mapped[&base].holes, 0xFF0F, "uncommitted prefix and suffix pages must stay holes");
            assert_eq!(s.mapped[&(base + TILE)].holes, 0xFFFF, "fully reserved tile must have all 16 holes set");

            // Reading back the full 2-tile span must not fault on uncommitted pages.
            s.readback_ranges(&[(base, 2 * TILE)]);

            // Now incrementally commit [base, base + 4*PAGE) and clear its hole bits (as `committed` does).
            assert!(!VirtualAlloc(base as *const c_void, 4 * PAGE, MEM_COMMIT, PAGE_READWRITE).is_null());
            let entry = s.mapped.get_mut(&base).unwrap();
            for i in 0..4 {
                entry.holes &= !(1u16 << i);
            }
            assert_eq!(entry.holes, 0xFF00);

            // Writing to the newly committed page is detected by `scan_page`.
            *(base as *mut u8) = 0x5A;
            let mut dirty = Vec::new();
            let page0 = std::slice::from_raw_parts(base as *const u8, PAGE);
            scan_page(page0, base, &mut entry.sent[0], &mut dirty);
            assert_eq!(dirty, vec![(base, CHUNK)]);

            s.wait_for(s.value);
            drop(s);
            assert_ne!(VirtualFree(base as *mut c_void, 0, MEM_RELEASE), 0);
        }
    }

    /// A tile decommitted and committed again before the next submit gets its pool tile back
    /// with no new tile mapping (cleared instead). One that is not is unmapped and freed by
    /// `flush_parked` (before a submit), or when the free list runs out; a placed buffer's
    /// release does not free it while it is parked.
    #[test]
    fn a_tile_decommitted_and_committed_again_keeps_its_video_memory() {
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
        use windows::Win32::Graphics::Dxgi::*;
        use windows_sys::Win32::System::Memory::{MEM_RELEASE, MEM_RESERVE};

        unsafe {
            let factory: IDXGIFactory4 = CreateDXGIFactory1().expect("DXGI factory");
            let mut dev: Option<ID3D12Device> = None;
            if D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
                let warp: IDXGIAdapter = factory.EnumWarpAdapter().expect("WARP");
                D3D12CreateDevice(&warp, D3D_FEATURE_LEVEL_11_0, &mut dev).expect("WARP D3D12 device");
            }
            let dev = dev.expect("D3D12 device");
            let base = VirtualAlloc(std::ptr::null(), 2 * TILE, MEM_RESERVE, PAGE_NOACCESS) as usize;
            assert!(base != 0 && base % TILE == 0);
            let mut reserved: Option<ID3D12Resource> = None;
            dev.CreateReservedResource(&buffer_desc((2 * TILE) as u64), D3D12_RESOURCE_STATE_COMMON, None, &mut reserved)
                .expect("CreateReservedResource");
            let reserved = reserved.expect("reserved buffer");
            let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).expect("queue");
            let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).expect("fence");
            let rb_fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).expect("rb_fence");
            let event = windows::Win32::System::Threading::CreateEventW(None, false, false, None).expect("event");
            let rb_alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).expect("rb_alloc");
            let rb_list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &rb_alloc, None).expect("rb_list");
            rb_list.Close().expect("rb_list close");

            let mut s = State {
                dev,
                queue,
                fence,
                value: 0,
                event,
                rb_fence,
                rb_value: 0,
                rb_queue_fences: FxHashMap::default(),
                rbq: None,
                rbq_value: 0,
                rb_jobs: VecDeque::new(),
                rb_pool: Vec::new(),
                imm_jobs: VecDeque::new(),
                job_seq: 0,
                windows: vec![(base, reserved)],
                pools: Vec::new(),
                free: Vec::new(),
                retired: Vec::new(),
                mapped: FxHashMap::default(),
                pinned: FxHashMap::default(),
                parked: FxHashMap::default(),
                quarantined: VecDeque::new(),
                queue_fences: FxHashMap::default(),
                kept: FxHashMap::default(),
                pending: Vec::new(),
                buffers: FxHashMap::default(),
                to_zero: BTreeSet::new(),
                staging: None,
                ring_write: 0,
                ring_submitted: 0,
                ring_free: 0,
                inflight: VecDeque::new(),
                zero_buf: None,
                slots: Vec::new(),
                next_slot: 0,
                readback: None,
                rb_alloc,
                rb_list,
                rb_last_fence: 0,
                waited: FxHashMap::default(),
                runs: BTreeMap::new(),
                scratch_runs: Vec::new(),
            };

            s.commit_range(base, TILE);
            let tile = s.mapped[&base].pool;
            assert_eq!(s.pending, vec![base], "a new tile's mapping waits for the next flush");
            assert_eq!(s.value, 0, "nothing sent yet");
            s.flush_pending();
            assert!(s.pending.is_empty());
            let v = s.value;
            assert!(v > 0, "a new tile is mapped");

            // decommitted: parked, still mapped, nothing sent to the GPU
            s.decommit_range(base, TILE);
            assert!(!s.mapped.contains_key(&base));
            assert_eq!(s.parked.get(&base), Some(&tile));
            assert!(!s.free.contains(&tile));
            assert_eq!(s.value, v, "no tile mapping changed");
            s.release_if_unused(base, tile);
            assert!(!s.free.contains(&tile), "a parked tile is not free");

            assert!(s.runs.is_empty() && runs_of(&s).is_empty());
            // committed again: the same tile, cleared, no new mapping
            s.to_zero.clear();
            let gen = mapped_gen();
            s.commit_range(base, TILE);
            assert_eq!(s.runs, BTreeMap::from([(base, base + TILE)]));
            assert_eq!(s.mapped[&base].pool, tile);
            assert_eq!(s.mapped[&base].holes, 0);
            assert!(s.parked.is_empty());
            assert_eq!(s.to_zero, BTreeSet::from([(base, TILE)]));
            assert_eq!(s.value, v, "no tile mapping changed");
            assert!(mapped_gen() > gen, "readers of `mapped` see it changed");

            // decommitted and flushed (as before a submit): unmapped, and free once the unmap is done
            s.decommit_range(base, TILE);
            assert!(s.flush_parked());
            assert!(s.parked.is_empty());
            s.wait_for(s.value);
            assert!(s.drain_quarantined());
            assert!(s.free.contains(&tile));
            assert!(s.value > v, "the unmapping is sent");
            assert!(!s.flush_parked(), "nothing left to flush");

            // no free tile left: the parked one is flushed and taken for the next commit once its
            // unmap is done
            s.commit_range(base, TILE);
            let tile = s.mapped[&base].pool;
            s.decommit_range(base, TILE);
            s.free.clear();
            assert!(s.flush_parked());
            s.wait_for(s.value);
            s.commit_range(base + TILE, TILE);
            assert!(s.parked.is_empty());
            assert_eq!(s.mapped[&(base + TILE)].pool, tile);
            assert!(!s.mapped.contains_key(&base));

            // a placed buffer uses the tile: the window keeps it when it is decommitted and
            // flushed, memory committed there again is the buffer's, and it is free only once
            // the buffer is gone and it was decommitted
            *s.pinned.entry(tile).or_default() += 1;
            let at = base + TILE;
            assert!(s.pending.contains(&at), "its mapping is not sent yet");
            s.decommit_range(at, TILE);
            assert!(!s.flush_parked(), "nothing to unmap");
            assert_eq!(s.kept.get(&at), Some(&tile));
            assert!(!s.free.contains(&tile));
            // kept before its mapping was sent: that stays pending, and the commit that takes the
            // tile back sends it
            let v = s.value;
            s.flush_pending();
            assert_eq!((s.value, &s.pending), (v, &vec![at]), "no mapping of a kept tile");
            s.commit_range(at, TILE);
            s.flush_pending();
            assert!(s.value > v && s.pending.is_empty(), "mapped with the commit");
            s.decommit_range(at, TILE);
            assert!(!s.flush_parked(), "nothing to unmap");
            assert_eq!(s.kept.get(&at), Some(&tile));
            let v = s.value;
            s.to_zero.clear();
            s.commit_range(at, TILE);
            assert_eq!(s.mapped[&at].pool, tile, "the buffer's tile");
            assert_eq!(s.to_zero, BTreeSet::from([(at, TILE)]));
            assert_eq!(s.value, v, "no tile mapping changed");
            s.decommit_range(at, TILE);
            assert!(!s.flush_parked());
            s.pinned.remove(&tile);
            s.release_if_unused(at, tile);
            assert!(s.kept.is_empty() && !s.free.contains(&tile));
            assert!(s.flush_parked());
            s.wait_for(s.value);
            assert!(s.drain_quarantined());
            assert!(s.free.contains(&tile));

            assert_eq!(s.runs, runs_of(&s));
            // committed and decommitted again before a flush: no mapping is sent for it
            s.flush_pending();
            s.flush_parked();
            let v = s.value;
            s.commit_range(base, TILE);
            assert_eq!(s.pending, vec![base]);
            s.decommit_range(base, TILE);
            s.flush_pending(); // as for an upload or readback before the title's next submit
            assert_eq!(s.value, v, "a tile gone before the flush is not mapped");
            assert_eq!(s.pending, vec![base], "parked, its mapping never sent: still pending");
            // taken back from `parked`: the flush now sends its mapping
            s.commit_range(base, TILE);
            s.flush_pending();
            assert!(s.pending.is_empty());
            assert!(s.value > v, "a tile parked before its mapping was sent is mapped");

            // neighbours are one run, a tile taken out of it leaves the other
            s.commit_range(base, 2 * TILE);
            assert_eq!(s.runs, BTreeMap::from([(base, base + 2 * TILE)]));
            s.decommit_range(base, TILE);
            assert_eq!(s.runs, BTreeMap::from([(base + TILE, base + 2 * TILE)]));
            s.decommit_range(base + TILE, PAGE);
            assert_eq!(s.runs, runs_of(&s), "a tile with pages left stays");
            s.flush_pending();

            s.wait_for(s.value);
            drop(s);
            assert_ne!(VirtualFree(base as *mut c_void, 0, MEM_RELEASE), 0);
        }
    }

    /// Sub-tile decommits (two 32 KiB halves, sixteen 4 KiB pages, or decommitting the last
    /// remaining committed page) must unmap the tile and return unpinned pool tiles to `free`
    /// once `holes` reaches `0xFFFF`, pruning stale `to_zero` entries for unmapped tiles.
    #[test]
    fn decommit_sub_tile_reclaims_pool_tile_when_all_pages_become_holes() {
        let mut mapped: FxHashMap<usize, TileEntry> = FxHashMap::default();
        let mut pinned: FxHashMap<(u32, u32), u32> = FxHashMap::default();
        let mut free: Vec<(u32, u32)> = Vec::new();
        let mut to_zero: BTreeSet<(usize, usize)> = BTreeSet::new();

        // 1. Two 32 KiB decommits across a full 64 KiB tile.
        let t0 = 0x10_0000usize;
        mapped.insert(t0, TileEntry::with_holes((0, 10), 0));
        assert!(decommit_tiles(&mut mapped, &pinned, &mut free, &mut to_zero, t0, TILE / 2).is_empty());
        assert_eq!(mapped[&t0].holes, 0x00FF);
        assert_eq!(to_zero.len(), 8);
        assert!(free.is_empty());

        let unmapped = decommit_tiles(&mut mapped, &pinned, &mut free, &mut to_zero, t0 + TILE / 2, TILE / 2);
        assert_eq!(unmapped, vec![t0]);
        assert!(!mapped.contains_key(&t0));
        assert_eq!(free, vec![(0, 10)]);
        assert!(to_zero.is_empty(), "stale to_zero entries for the unmapped tile must be pruned");

        // 2. Sixteen individual 4 KiB decommits in reverse order.
        let t1 = 0x20_0000usize;
        free.clear();
        mapped.insert(t1, TileEntry::with_holes((0, 11), 0));
        for i in (1..PAGES_PER_TILE).rev() {
            let u = decommit_tiles(&mut mapped, &pinned, &mut free, &mut to_zero, t1 + i * PAGE, PAGE);
            assert!(u.is_empty(), "tile still has committed pages after step {i}");
        }
        assert!(free.is_empty());
        let u = decommit_tiles(&mut mapped, &pinned, &mut free, &mut to_zero, t1, PAGE);
        assert_eq!(u, vec![t1]);
        assert!(!mapped.contains_key(&t1));
        assert_eq!(free, vec![(0, 11)]);
        assert!(to_zero.is_empty());

        // 3. Pinned tile decommitted in two halves: unmapped from `mapped`, NOT returned to `free`.
        let t2 = 0x30_0000usize;
        free.clear();
        mapped.insert(t2, TileEntry::with_holes((0, 12), 0));
        pinned.insert((0, 12), 1);
        assert!(decommit_tiles(&mut mapped, &pinned, &mut free, &mut to_zero, t2, TILE / 2).is_empty());
        let u = decommit_tiles(&mut mapped, &pinned, &mut free, &mut to_zero, t2 + TILE / 2, TILE / 2);
        assert_eq!(u, vec![t2]);
        assert!(!mapped.contains_key(&t2));
        assert!(free.is_empty(), "pinned pool tile must not return to free list");
        assert!(to_zero.is_empty());

        // 4. Single-committed-page tile decommit reclaims tile without dropping another tile's to_zero.
        let t3 = 0x40_0000usize;
        let t4 = 0x50_0000usize;
        mapped.insert(t3, TileEntry::with_holes((0, 13), 0xFFFF & !(1u16 << 2)));
        mapped.insert(t4, TileEntry::with_holes((0, 14), 0));
        assert!(decommit_tiles(&mut mapped, &pinned, &mut free, &mut to_zero, t4, PAGE).is_empty());
        assert_eq!(to_zero, BTreeSet::from([(t4, PAGE)]));
        let u = decommit_tiles(&mut mapped, &pinned, &mut free, &mut to_zero, t3 + 2 * PAGE, PAGE);
        assert_eq!(u, vec![t3]);
        assert_eq!(free, vec![(0, 13)]);
        assert_eq!(to_zero, BTreeSet::from([(t4, PAGE)]), "unrelated mapped tile's to_zero must remain");

        // 5. The pieces as runs: neighbours together, pieces inside a whole tile with it.
        let t5 = 0x60_0000usize;
        to_zero.extend([(t5, TILE), (t5 + 3 * PAGE, PAGE), (t5 + TILE, TILE), (t5 + 2 * TILE + PAGE, PAGE)]);
        assert_eq!(zero_runs(to_zero), vec![(t4, PAGE), (t5, 2 * TILE), (t5 + 2 * TILE + PAGE, PAGE)]);
    }

    /// The pages in `early`, checking that its ranges are page-aligned, not empty, disjoint and
    /// merged (no two touch).
    fn early_pages(early: &BTreeMap<usize, usize>) -> std::collections::BTreeSet<usize> {
        let mut last_end = None;
        let mut pages = std::collections::BTreeSet::new();
        for (&a, &b) in early {
            assert!(a % PAGE == 0 && b % PAGE == 0 && a < b, "range {a:#x}..{b:#x}");
            assert!(last_end.is_none_or(|e| e < a), "range {a:#x}..{b:#x} touches the one before");
            last_end = Some(b);
            pages.extend((a..b).step_by(PAGE));
        }
        pages
    }

    /// xorshift64*: the same sequence on every run
    fn next(x: &mut u64) -> u64 {
        *x ^= *x >> 12;
        *x ^= *x << 25;
        *x ^= *x >> 27;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Before the device exists, commits and decommits leave the pages a plain page set would
    /// hold, with the live path's rule (a range counts every page it touches): random ranges,
    /// unaligned too, over 300 pages.
    #[test]
    fn early_commits_are_the_pages_a_page_set_holds() {
        let base = 0x4_2000_0000usize;
        for seed in 1..=8u64 {
            let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
            let (mut early, mut model) = (BTreeMap::new(), std::collections::BTreeSet::new());
            for _ in 0..4000 {
                let a = base + (next(&mut x) % (300 * PAGE as u64)) as usize;
                let n = match next(&mut x) % 4 {
                    0 => PAGE,
                    1 => 1 + (next(&mut x) % 64) as usize,
                    _ => 1 + (next(&mut x) % (24 * PAGE as u64)) as usize,
                };
                let (start, end) = page_span(a, n);
                let pages = (start..end).step_by(PAGE);
                if next(&mut x) % 3 == 0 {
                    early_remove(&mut early, start, end);
                    pages.for_each(|p| {
                        model.remove(&p);
                    });
                } else {
                    early_add(&mut early, start, end);
                    model.extend(pages);
                }
                assert_eq!(early_pages(&early), model, "seed {seed}");
            }
        }
    }

    /// The same pages as the list this replaced, for what that list handled exactly: one
    /// entry per committed page (`XMemMapPhysicalPagesScatter`, `home_again`) and page-aligned
    /// decommits of any length.
    #[test]
    fn early_commits_match_the_old_list_for_page_entries() {
        let base = 0x4_2000_0000usize;
        let mut x = 0x1234_5678_9ABC_DEF1u64;
        let (mut early, mut old): (BTreeMap<usize, usize>, Vec<(usize, usize)>) = (BTreeMap::new(), Vec::new());
        for _ in 0..20000 {
            let a = base + (next(&mut x) % 512) as usize * PAGE;
            if next(&mut x) % 2 == 0 {
                old.push((a, PAGE));
                early_add(&mut early, a, a + PAGE);
            } else {
                let n = (1 + next(&mut x) % 8) as usize * PAGE;
                old.retain(|&(y, m)| y + m <= a || y >= a + n);
                early_remove(&mut early, a, a + n);
            }
        }
        let old_pages: std::collections::BTreeSet<usize> = old.iter().map(|&(y, _)| y).collect();
        assert_eq!(early_pages(&early), old_pages);
    }

    /// A range committed in one call keeps its other pages when one page of it is decommitted
    /// (`commit_homes` commits several pages, `move_home_away` gives one back). The list this
    /// replaced dropped the whole entry, so `init` left those still-committed pages as holes.
    #[test]
    fn early_decommit_of_one_page_keeps_the_rest() {
        let a = 0x4_2000_0000usize;
        let mut early = BTreeMap::new();
        early_add(&mut early, a, a + 4 * TILE);
        early_remove(&mut early, a + TILE, a + 2 * TILE);
        assert_eq!(early.into_iter().collect::<Vec<_>>(), vec![(a, a + TILE), (a + 2 * TILE, a + 4 * TILE)]);
    }

    /// Time for n 64 KiB pages committed one at a time and decommitted one at a time in
    /// reverse, before the device exists (`cargo test --release -p xcl --lib --
    /// --ignored --exact hle::d3d12x::sparse::tests::early_commit_timing --nocapture`,
    /// `XCL_EARLY_PAGES=16384,32768`). Run it alone: a test that made the device keeps a state.
    #[test]
    #[ignore]
    fn early_commit_timing() {
        if STATE.lock().unwrap().is_some() {
            return eprintln!("early_commit_timing: a sparse state exists – run it alone");
        }
        let sizes: Vec<usize> = std::env::var("XCL_EARLY_PAGES").ok().map_or(vec![4096, 8192, 16384, 32768], |v| v.split(',').map(|s| s.trim().parse().unwrap()).collect());
        let base = 0x4_2000_0000usize;
        for n in sizes {
            let t = std::time::Instant::now();
            for i in 0..n {
                unsafe { committed(base + i * TILE, TILE) };
            }
            let commit = t.elapsed();
            let t = std::time::Instant::now();
            for i in (0..n).rev() {
                unsafe { decommitted(base + i * TILE, TILE) };
            }
            let decommit = t.elapsed();
            println!("early: {n} pages: commit {:.1} ms, decommit {:.1} ms", commit.as_secs_f64() * 1e3, decommit.as_secs_f64() * 1e3);
        }
    }
}

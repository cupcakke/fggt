const std = @import("std");
const mem = std.mem;
const math = std.math;
const Allocator = mem.Allocator;
const ArrayList = std.ArrayList;
const StringHashMap = std.StringHashMap;
const Sha256 = std.crypto.hash.sha2.Sha256;
const Complex = std.math.Complex;

pub const Error = error{
    OutOfBounds,
    ShapeMismatch,
};

pub const Tensor = struct {
    data: []f32,
    shape: struct { dims: []const usize },
    allocator: Allocator,

    pub fn init(allocator: Allocator, dims: []const usize) !Tensor {
        const dims_copy = try allocator.dupe(usize, dims);
        errdefer allocator.free(dims_copy);
        var elements: usize = 1;
        for (dims_copy) |dim| {
            elements = std.math.mul(usize, elements, dim) catch return error.Overflow;
        }
        const data = try allocator.alloc(f32, elements);
        return Tensor{
            .data = data,
            .shape = .{ .dims = dims_copy },
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Tensor) void {
        self.allocator.free(self.data);
        self.allocator.free(self.shape.dims);
        self.data = &[_]f32{};
        self.shape.dims = &[_]usize{};
    }

    pub fn elementCount(self: *const Tensor) usize {
        return self.data.len;
    }
};

pub const BitSet = struct {
    bits: []u64,
    allocator: Allocator,

    pub fn init(allocator: Allocator, num_bits: usize) !BitSet {
        const words = (num_bits + 63) / 64;
        const bits = try allocator.alloc(u64, words);
        @memset(bits, 0);
        return BitSet{ .bits = bits, .allocator = allocator };
    }

    pub fn deinit(self: *BitSet) void {
        self.allocator.free(self.bits);
        self.bits = &[_]u64{};
    }

    fn growFor(self: *BitSet, index: usize) !void {
        const needed_words = (index / 64) + 1;
        if (needed_words <= self.bits.len) return;
        var new_words = self.bits.len;
        if (new_words == 0) new_words = 1;
        while (new_words < needed_words) new_words *= 2;
        const grown = try self.allocator.realloc(self.bits, new_words);
        @memset(grown[self.bits.len..], 0);
        self.bits = grown;
    }

    pub fn set(self: *BitSet, index: usize) void {
        std.debug.assert(index / 64 < self.bits.len);
        self.bits[index / 64] |= @as(u64, 1) << @intCast(index % 64);
    }

    pub fn setGrow(self: *BitSet, index: usize) !void {
        try self.growFor(index);
        self.bits[index / 64] |= @as(u64, 1) << @intCast(index % 64);
    }

    pub fn unset(self: *BitSet, index: usize) void {
        if (index / 64 >= self.bits.len) return;
        self.bits[index / 64] &= ~(@as(u64, 1) << @intCast(index % 64));
    }

    pub fn get(self: *const BitSet, index: usize) bool {
        if (index / 64 >= self.bits.len) return false;
        return (self.bits[index / 64] & (@as(u64, 1) << @intCast(index % 64))) != 0;
    }

    pub fn intersectionPopCount(self: *const BitSet, other: *const BitSet) usize {
        const words = @min(self.bits.len, other.bits.len);
        var count: usize = 0;
        var i: usize = 0;
        while (i < words) : (i += 1) {
            count += @popCount(self.bits[i] & other.bits[i]);
        }
        return count;
    }

    pub fn unionPopCount(self: *const BitSet, other: *const BitSet) usize {
        const words = @max(self.bits.len, other.bits.len);
        var count: usize = 0;
        var i: usize = 0;
        while (i < words) : (i += 1) {
            const a: u64 = if (i < self.bits.len) self.bits[i] else 0;
            const b: u64 = if (i < other.bits.len) other.bits[i] else 0;
            count += @popCount(a | b);
        }
        return count;
    }

    pub fn jaccardEstimate(self: *const BitSet, other: *const BitSet) f32 {
        const union_count = self.unionPopCount(other);
        if (union_count == 0) return 1.0;
        return @as(f32, @floatFromInt(self.intersectionPopCount(other))) / @as(f32, @floatFromInt(union_count));
    }
};

pub const RankedSegment = struct {
    tokens: []u32,
    position: u64,
    score: f32,
    anchor: bool,

    pub fn init(allocator: Allocator, tokens: []const u32, score: f32, position: u64, anchor: bool) !RankedSegment {
        return RankedSegment{
            .tokens = try allocator.dupe(u32, tokens),
            .score = score,
            .position = position,
            .anchor = anchor,
        };
    }

    pub fn deinit(self: *RankedSegment, allocator: Allocator) void {
        allocator.free(self.tokens);
        self.tokens = &[_]u32{};
    }
};

pub fn stableHash(data: []const u8, seed: u64) u64 {
    var h = seed ^ 0xcbf29ce484222325;
    for (data) |byte| {
        h ^= byte;
        h *%= 0x100000001b3;
    }
    h ^= h >> 30;
    h *%= 0xbf58476d1ce4e5b9;
    h ^= h >> 27;
    h *%= 0x94d049bb133111eb;
    h ^= h >> 31;
    return h;
}

pub fn createFilePath(path: []const u8, flags: std.fs.File.CreateFlags) !std.fs.File {
    return std.fs.cwd().createFile(path, flags);
}

pub fn openFilePath(path: []const u8, flags: std.fs.File.OpenFlags) !std.fs.File {
    return std.fs.cwd().openFile(path, flags);
}

pub const ArenaAllocator = struct {
    inner: std.heap.ArenaAllocator,

    pub fn init(child_allocator: Allocator) ArenaAllocator {
        return ArenaAllocator{ .inner = std.heap.ArenaAllocator.init(child_allocator) };
    }

    pub fn allocator(self: *ArenaAllocator) Allocator {
        return self.inner.allocator();
    }

    pub fn reset(self: *ArenaAllocator) void {
        _ = self.inner.reset(.retain_capacity);
    }

    pub fn deinit(self: *ArenaAllocator) void {
        self.inner.deinit();
    }
};

pub const PoolAllocator = struct {
    inner: std.heap.GeneralPurposeAllocator(.{
        .enable_memory_limit = true,
        .retain_metadata = true,
    }),

    pub fn init(_: Allocator) PoolAllocator {
        return PoolAllocator{ .inner = .{} };
    }

    pub fn allocator(self: *PoolAllocator) Allocator {
        return self.inner.allocator();
    }

    pub fn deinit(self: *PoolAllocator) void {
        _ = self.inner.deinit();
    }
};

pub const BuddyAllocator = struct {
    inner: std.heap.GeneralPurposeAllocator(.{}),

    pub fn init(_: Allocator) BuddyAllocator {
        return BuddyAllocator{ .inner = .{} };
    }

    pub fn allocator(self: *BuddyAllocator) Allocator {
        return self.inner.allocator();
    }

    pub fn deinit(self: *BuddyAllocator) void {
        _ = self.inner.deinit();
    }
};

pub const EdgeQuality = enum(u8) {
    coherent,
    decoherent,

    pub fn toString(self: EdgeQuality) []const u8 {
        return switch (self) {
            .coherent => "coherent",
            .decoherent => "decoherent",
        };
    }
};

pub const Node = struct {
    id: []u8,
    label: []u8,
    quantum_state: Complex(f64),
    phase: f64,
    metadata: StringHashMap([]u8),
    allocator: Allocator,

    pub fn init(allocator: Allocator, id: []const u8, label: []const u8) !Node {
        return initWithComplex(allocator, id, label, Complex(f64).init(1.0, 0.0), 0.0);
    }

    pub fn initWithComplex(allocator: Allocator, id: []const u8, label: []const u8, quantum_state: Complex(f64), phase: f64) !Node {
        const id_copy = try allocator.dupe(u8, id);
        errdefer allocator.free(id_copy);
        const label_copy = try allocator.dupe(u8, label);
        errdefer allocator.free(label_copy);
        return Node{
            .id = id_copy,
            .label = label_copy,
            .quantum_state = quantum_state,
            .phase = phase,
            .metadata = StringHashMap([]u8).init(allocator),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Node) void {
        self.allocator.free(self.id);
        self.allocator.free(self.label);
        var it = self.metadata.iterator();
        while (it.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            self.allocator.free(entry.value_ptr.*);
        }
        self.metadata.deinit();
        self.id = &[_]u8{};
        self.label = &[_]u8{};
    }

    pub fn setMetadata(self: *Node, key: []const u8, value: []const u8) !void {
        const value_copy = try self.allocator.dupe(u8, value);
        errdefer self.allocator.free(value_copy);
        if (self.metadata.getPtr(key)) |existing| {
            self.allocator.free(existing.*);
            existing.* = value_copy;
            return;
        }
        const key_copy = try self.allocator.dupe(u8, key);
        errdefer self.allocator.free(key_copy);
        try self.metadata.put(key_copy, value_copy);
    }

    pub fn getMetadata(self: *const Node, key: []const u8) ?[]const u8 {
        return self.metadata.get(key);
    }
};

pub const Edge = struct {
    source: []u8,
    target: []u8,
    quality: EdgeQuality,
    weight: f64,
    quantum_state: Complex(f64),
    frequency: f64,
    metadata: StringHashMap([]u8),
    allocator: Allocator,

    pub fn init(allocator: Allocator, source: []const u8, target: []const u8, quality: EdgeQuality, weight: f64) !Edge {
        return initWithComplex(allocator, source, target, quality, weight, Complex(f64).init(1.0, 0.0), 0.0);
    }

    pub fn initWithComplex(allocator: Allocator, source: []const u8, target: []const u8, quality: EdgeQuality, weight: f64, quantum_state: Complex(f64), frequency: f64) !Edge {
        const source_copy = try allocator.dupe(u8, source);
        errdefer allocator.free(source_copy);
        const target_copy = try allocator.dupe(u8, target);
        errdefer allocator.free(target_copy);
        return Edge{
            .source = source_copy,
            .target = target_copy,
            .quality = quality,
            .weight = weight,
            .quantum_state = quantum_state,
            .frequency = frequency,
            .metadata = StringHashMap([]u8).init(allocator),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Edge) void {
        self.allocator.free(self.source);
        self.allocator.free(self.target);
        var it = self.metadata.iterator();
        while (it.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            self.allocator.free(entry.value_ptr.*);
        }
        self.metadata.deinit();
        self.source = &[_]u8{};
        self.target = &[_]u8{};
    }

    pub fn setMetadata(self: *Edge, key: []const u8, value: []const u8) !void {
        const value_copy = try self.allocator.dupe(u8, value);
        errdefer self.allocator.free(value_copy);
        if (self.metadata.getPtr(key)) |existing| {
            self.allocator.free(existing.*);
            existing.* = value_copy;
            return;
        }
        const key_copy = try self.allocator.dupe(u8, key);
        errdefer self.allocator.free(key_copy);
        try self.metadata.put(key_copy, value_copy);
    }

    pub fn getMetadata(self: *const Edge, key: []const u8) ?[]const u8 {
        return self.metadata.get(key);
    }
};

pub const SelfSimilarRelationalGraph = struct {
    nodes: ArrayList(Node),
    edges: ArrayList(Edge),
    allocator: Allocator,

    pub fn init(allocator: Allocator) SelfSimilarRelationalGraph {
        return SelfSimilarRelationalGraph{
            .nodes = ArrayList(Node).init(allocator),
            .edges = ArrayList(Edge).init(allocator),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *SelfSimilarRelationalGraph) void {
        for (self.nodes.items) |*node| node.deinit();
        self.nodes.deinit();
        for (self.edges.items) |*edge| edge.deinit();
        self.edges.deinit();
    }

    pub fn addNode(self: *SelfSimilarRelationalGraph, node: Node) !void {
        try self.nodes.append(node);
    }

    pub fn addEdge(self: *SelfSimilarRelationalGraph, edge: Edge) !void {
        try self.edges.append(edge);
    }

    pub fn nodeCount(self: *const SelfSimilarRelationalGraph) usize {
        return self.nodes.items.len;
    }

    pub fn edgeCount(self: *const SelfSimilarRelationalGraph) usize {
        return self.edges.items.len;
    }

    pub fn findNodeById(self: *const SelfSimilarRelationalGraph, id: []const u8) ?*Node {
        for (self.nodes.items) |*node| {
            if (std.mem.eql(u8, node.id, id)) return node;
        }
        return null;
    }

    pub fn coherenceRatio(self: *const SelfSimilarRelationalGraph) f64 {
        if (self.edges.items.len == 0) return 1.0;
        var coherent: usize = 0;
        for (self.edges.items) |edge| {
            if (edge.quality == .coherent) coherent += 1;
        }
        return @as(f64, @floatFromInt(coherent)) / @as(f64, @floatFromInt(self.edges.items.len));
    }
};

pub const ChaosCoreKernel = struct {
    allocator: Allocator,
    blocks: ArrayList(MemoryBlock),
    graph: SelfSimilarRelationalGraph,
    mutex: std.Thread.Mutex,

    pub const MemoryBlock = struct {
        data: []u8,
        tag: ?[]u8,

        fn deinit(self: *MemoryBlock, allocator: Allocator) void {
            allocator.free(self.data);
            if (self.tag) |t| allocator.free(t);
            self.data = &[_]u8{};
            self.tag = null;
        }
    };

    pub fn init(allocator: Allocator) ChaosCoreKernel {
        return ChaosCoreKernel{
            .allocator = allocator,
            .blocks = ArrayList(MemoryBlock).init(allocator),
            .graph = SelfSimilarRelationalGraph.init(allocator),
            .mutex = .{},
        };
    }

    pub fn deinit(self: *ChaosCoreKernel) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        for (self.blocks.items) |*block| block.deinit(self.allocator);
        self.blocks.deinit();
        self.graph.deinit();
    }

    pub fn allocateMemory(self: *ChaosCoreKernel, data: []const u8, tag: ?[]const u8) !usize {
        self.mutex.lock();
        defer self.mutex.unlock();
        const data_copy = try self.allocator.dupe(u8, data);
        errdefer self.allocator.free(data_copy);
        var tag_copy: ?[]u8 = null;
        if (tag) |t| {
            tag_copy = try self.allocator.dupe(u8, t);
        }
        errdefer if (tag_copy) |tc| self.allocator.free(tc);
        try self.blocks.append(.{ .data = data_copy, .tag = tag_copy });
        return self.blocks.items.len - 1;
    }

    pub fn getMemory(self: *ChaosCoreKernel, index: usize) ?[]const u8 {
        self.mutex.lock();
        defer self.mutex.unlock();
        if (index >= self.blocks.items.len) return null;
        return self.blocks.items[index].data;
    }

    pub fn memoryCount(self: *ChaosCoreKernel) usize {
        self.mutex.lock();
        defer self.mutex.unlock();
        return self.blocks.items.len;
    }

    pub fn synchronizeGraphWithMemory(self: *ChaosCoreKernel) !usize {
        self.mutex.lock();
        defer self.mutex.unlock();
        var synchronized: usize = 0;
        for (self.blocks.items, 0..) |block, index| {
            var id_buf: [32]u8 = undefined;
            const id_text = std.fmt.bufPrint(id_buf[0..], "mem_{d}", .{index}) catch continue;
            if (self.graph.findNodeById(id_text) != null) continue;
            var node = try Node.init(self.allocator, id_text, block.data);
            errdefer node.deinit();
            if (block.tag) |t| {
                try node.setMetadata("tag", t);
            }
            try self.graph.addNode(node);
            synchronized += 1;
        }
        return synchronized;
    }
};


pub const MGT = struct {
    token_to_id: std.StringHashMap(u32),
    id_to_token: std.AutoHashMap(u32, []const u8),
    prefixes: std.StringHashMap(u32),
    suffixes: std.StringHashMap(u32),
    roots: std.StringHashMap(u32),
    bpe_pairs: std.AutoHashMap(u64, BPEMerge),
    anchors: std.StringHashMap(u64),
    allocated_strings: std.ArrayList([]u8),
    allocator: Allocator,
    next_token_id: u32,
    language: Language,
    max_vocab_size: ?u32,
    allow_dynamic_tokens: bool,
    sorted_prefix_keys: std.ArrayList([]const u8),
    sorted_suffix_keys: std.ArrayList([]const u8),
    max_prefix_len: usize,
    max_suffix_len: usize,
    byte_token_ids: [256]?u32,
    byte_token_values: std.AutoHashMap(u32, u8),

    pub const Language = enum {
        english,
        hungarian,
        dual,
    };

    const BPEMerge = struct {
        token_id: u32,
        priority: u32,
    };

    const TokenPairKey = struct {
        first: u32,
        second: u32,
    };

    fn packPair(first: u32, second: u32) u64 {
        return (@as(u64, first) << 32) | @as(u64, second);
    }

    const PairFreq = struct {
        key: TokenPairKey,
        freq: u64,
    };

    const BpeSequence = struct {
        storage: []u32,
        len: usize,

        fn items(self: *const BpeSequence) []const u32 {
            return self.storage[0..self.len];
        }

        fn mutableItems(self: *BpeSequence) []u32 {
            return self.storage[0..self.len];
        }
    };

    const BpeScanWorkerCtx = struct {
        sequences: []BpeSequence,
        pair_freqs: std.AutoHashMap(TokenPairKey, u64),
        err: ?anyerror,
    };

    const BpeRebuildWorkerCtx = struct {
        sequences: []BpeSequence,
        best_first: u32,
        best_second: u32,
        merge_id: u32,
    };

    const SPECIAL_TOKENS = struct {
        const PAD: u32 = 0;
        const UNK: u32 = 1;
        const BOS: u32 = 2;
        const EOS: u32 = 3;
    };

    pub fn init(allocator: Allocator, vocab: []const []const u8, anchors: []const []const u8, max_vocab_size: ?u32, language: Language) !MGT {
        if (max_vocab_size) |max| {
            if (max < 4) return error.VocabularyTooSmall;
        }

        var mgt = initEmpty(allocator, max_vocab_size, language);
        errdefer mgt.deinit();

        _ = try mgt.addToken("[PAD]");
        _ = try mgt.addToken("[UNK]");
        _ = try mgt.addToken("[BOS]");
        _ = try mgt.addToken("[EOS]");

        for (vocab) |word| {
            if (!mgt.canAddToken()) break;
            _ = try mgt.addToken(word);
        }

        try mgt.initMorphemes();

        for (anchors) |anch| {
            if (!mgt.canAddToken() and !mgt.token_to_id.contains(anch)) break;
            const tid = mgt.token_to_id.get(anch) orelse try mgt.addToken(anch);
            const key = mgt.id_to_token.get(tid) orelse return error.InvalidData;
            try mgt.anchors.put(key, @as(u64, tid));
        }

        return mgt;
    }

    pub fn initWithArena(arena: *ArenaAllocator, vocab: []const []const u8, anchors_list: []const []const u8, max_vocab_size: ?u32, language: Language) !MGT {
        return init(arena.allocator(), vocab, anchors_list, max_vocab_size, language);
    }

    pub fn initWithPool(pool: *PoolAllocator, vocab: []const []const u8, anchors_list: []const []const u8, max_vocab_size: ?u32, language: Language) !MGT {
        return init(pool.allocator(), vocab, anchors_list, max_vocab_size, language);
    }

    pub fn initWithBuddy(buddy: *BuddyAllocator, vocab: []const []const u8, anchors_list: []const []const u8, max_vocab_size: ?u32, language: Language) !MGT {
        return init(buddy.allocator(), vocab, anchors_list, max_vocab_size, language);
    }

    fn initEmpty(allocator: Allocator, max_vocab_size: ?u32, language: Language) MGT {
        return .{
            .token_to_id = std.StringHashMap(u32).init(allocator),
            .id_to_token = std.AutoHashMap(u32, []const u8).init(allocator),
            .prefixes = std.StringHashMap(u32).init(allocator),
            .suffixes = std.StringHashMap(u32).init(allocator),
            .roots = std.StringHashMap(u32).init(allocator),
            .bpe_pairs = std.AutoHashMap(u64, BPEMerge).init(allocator),
            .anchors = std.StringHashMap(u64).init(allocator),
            .allocated_strings = std.ArrayList([]u8).init(allocator),
            .allocator = allocator,
            .next_token_id = 0,
            .language = language,
            .max_vocab_size = max_vocab_size,
            .allow_dynamic_tokens = true,
            .sorted_prefix_keys = std.ArrayList([]const u8).init(allocator),
            .sorted_suffix_keys = std.ArrayList([]const u8).init(allocator),
            .max_prefix_len = 0,
            .max_suffix_len = 0,
            .byte_token_ids = [_]?u32{null} ** 256,
            .byte_token_values = std.AutoHashMap(u32, u8).init(allocator),
        };
    }

    fn canAddToken(self: *const MGT) bool {
        if (self.next_token_id == std.math.maxInt(u32)) return false;
        if (self.max_vocab_size) |max| {
            return self.token_to_id.count() < @as(usize, max);
        }
        return true;
    }

    fn reset(self: *MGT) void {
        const allocator = self.allocator;
        const mvs = self.max_vocab_size;
        const lang = self.language;
        self.deinit();
        self.* = initEmpty(allocator, mvs, lang);
    }

    fn initMorphemes(self: *MGT) !void {
        const english_prefix_list = [_][]const u8{
            "un",  "re",   "pre",   "dis",   "mis",  "over", "under", "out",
            "sub", "inter", "fore",  "de",    "trans", "super", "semi", "anti",
            "mid", "non",   "ex",    "post",  "pro",  "co",    "en",   "em",
        };

        const hungarian_prefix_list = [_][]const u8{
            "meg", "el", "fel", "le", "be", "ki", "rá", "át", "szét", "vissza",
            "ide", "oda", "alá", "fölé", "közé", "egy", "össze", "tul", "hozzá", "körül",
            "alig", "éppen", "majd", "csak", "is", "leg", "legesleg",
        };

        const english_suffix_list = [_][]const u8{
            "ing", "ed",  "er",   "est",  "ly",   "tion", "sion", "ness",
            "ment", "ful", "less", "ous",  "ive",  "able", "ible", "al",
            "ial", "y",   "s",    "es",   "en",   "ize",  "ise",  "ate",
        };

        const hungarian_suffix_list = [_][]const u8{
            "ság", "ség", "ságú", "ségű", "é", "je", "ja", "ban", "ben",
            "ba", "be", "ból", "ből", "hoz", "hez", "höz", "tól", "től",
            "nak", "nek", "val", "vel", "ért", "ul", "ül", "ként", "án",
            "én", "ig", "at", "et", "tat", "tet", "ott", "ett", "atlan",
            "etlen", "talan", "telen", "ál", "él", "oz", "ez", "öd", "ed",
            "gyet", "get", "j", "unk", "jatok", "játok", "i", "ni", "nként",
            "kor", "ra", "re",
        };

        const prefix_lists: [2][]const []const u8 = switch (self.language) {
            .english => .{ english_prefix_list[0..], english_prefix_list[0..0] },
            .hungarian => .{ hungarian_prefix_list[0..], hungarian_prefix_list[0..0] },
            .dual => .{ english_prefix_list[0..], hungarian_prefix_list[0..] },
        };

        const suffix_lists: [2][]const []const u8 = switch (self.language) {
            .english => .{ english_suffix_list[0..], english_suffix_list[0..0] },
            .hungarian => .{ hungarian_suffix_list[0..], hungarian_suffix_list[0..0] },
            .dual => .{ english_suffix_list[0..], hungarian_suffix_list[0..] },
        };

        for (prefix_lists) |prefix_list| {
            for (prefix_list) |prefix| {
                if (!self.canAddToken() and !self.token_to_id.contains(prefix)) break;
                const id = self.token_to_id.get(prefix) orelse try self.addToken(prefix);
                const key = self.id_to_token.get(id) orelse return error.InvalidData;
                try self.prefixes.put(key, id);
            }
        }

        for (suffix_lists) |suffix_list| {
            for (suffix_list) |suffix| {
                if (!self.canAddToken() and !self.token_to_id.contains(suffix)) break;
                const id = self.token_to_id.get(suffix) orelse try self.addToken(suffix);
                const key = self.id_to_token.get(id) orelse return error.InvalidData;
                try self.suffixes.put(key, id);
            }
        }

        try self.rebuildSortedMorphemes();
    }

    fn rebuildSortedMorphemes(self: *MGT) !void {
        self.sorted_prefix_keys.clearRetainingCapacity();
        var pit = self.prefixes.iterator();
        while (pit.next()) |entry| {
            try self.sorted_prefix_keys.append(entry.key_ptr.*);
        }

        std.mem.sort([]const u8, self.sorted_prefix_keys.items, {}, struct {
            fn lessThan(_: void, a: []const u8, b: []const u8) bool {
                return std.mem.lessThan(u8, a, b);
            }
        }.lessThan);

        self.max_prefix_len = 0;
        for (self.sorted_prefix_keys.items) |key| {
            self.max_prefix_len = @max(self.max_prefix_len, key.len);
        }

        self.sorted_suffix_keys.clearRetainingCapacity();
        var sit = self.suffixes.iterator();
        while (sit.next()) |entry| {
            try self.sorted_suffix_keys.append(entry.key_ptr.*);
        }

        std.mem.sort([]const u8, self.sorted_suffix_keys.items, {}, struct {
            fn lessThan(_: void, a: []const u8, b: []const u8) bool {
                return std.mem.lessThan(u8, a, b);
            }
        }.lessThan);

        self.max_suffix_len = 0;
        for (self.sorted_suffix_keys.items) |key| {
            self.max_suffix_len = @max(self.max_suffix_len, key.len);
        }
    }

    pub fn deinit(self: *MGT) void {
        self.token_to_id.deinit();
        self.id_to_token.deinit();
        self.prefixes.deinit();
        self.suffixes.deinit();
        self.roots.deinit();
        self.bpe_pairs.deinit();
        self.anchors.deinit();
        self.sorted_prefix_keys.deinit();
        self.sorted_suffix_keys.deinit();
        self.byte_token_values.deinit();

        for (self.allocated_strings.items) |str| {
            self.allocator.free(str);
        }

        self.allocated_strings.deinit();
    }

    fn isWhitespace(c: u8) bool {
        return c == ' ' or c == '\n' or c == '\t' or c == '\r';
    }

    fn isPunctuation(c: u8) bool {
        return c == '.' or c == ',' or c == '!' or c == '?' or c == ';' or
            c == ':' or c == '"' or c == '\'' or c == '(' or c == ')' or
            c == '{' or c == '}';
    }

    fn isKnownSpecialTokenStart(self: *const MGT, text: []const u8, pos: usize) bool {
        if (pos >= text.len or text[pos] != '[') return false;

        const specials = [_][]const u8{ "[PAD]", "[UNK]", "[BOS]", "[EOS]" };
        for (specials) |special| {
            if (special.len <= text.len - pos and
                mem.eql(u8, text[pos .. pos + special.len], special) and
                self.token_to_id.contains(special))
            {
                return true;
            }
        }

        return false;
    }

    fn getKnownSpecialTokenLen(self: *const MGT, text: []const u8, pos: usize) ?usize {
        if (pos >= text.len or text[pos] != '[') return null;

        const specials = [_][]const u8{ "[PAD]", "[UNK]", "[BOS]", "[EOS]" };
        for (specials) |special| {
            if (special.len <= text.len - pos and
                mem.eql(u8, text[pos .. pos + special.len], special) and
                self.token_to_id.contains(special))
            {
                return special.len;
            }
        }

        return null;
    }

    fn utf8CharLen(first_byte: u8) u8 {
        if (first_byte & 0x80 == 0) return 1;
        if (first_byte & 0xE0 == 0xC0) return 2;
        if (first_byte & 0xF0 == 0xE0) return 3;
        if (first_byte & 0xF8 == 0xF0) return 4;
        return 1;
    }

    fn safeUtf8SequenceLenAt(text: []const u8, pos: usize) usize {
        if (pos >= text.len) return 0;

        const sequence_len: usize = utf8CharLen(text[pos]);
        if (sequence_len > text.len - pos) return 1;
        if (sequence_len == 1) return 1;

        var i: usize = 1;
        while (i < sequence_len) : (i += 1) {
            if ((text[pos + i] & 0xC0) != 0x80) return 1;
        }

        return sequence_len;
    }

    fn truncateUtf8Input(text: []const u8, maximum_bytes: usize) []const u8 {
        const length = @min(text.len, maximum_bytes);
        var boundary = length;
        while (boundary > 0 and boundary < text.len and (text[boundary] & 0xC0) == 0x80) {
            boundary -= 1;
        }
        return text[0..boundary];
    }

    fn emitToken(self: *const MGT, tid: u32, byte_pos: usize, out_tokens: *std.ArrayList(u32), out_anchors: ?*std.ArrayList(usize)) !void {
        try out_tokens.append(tid);

        if (out_anchors) |anchors_out| {
            if (self.id_to_token.get(tid)) |token_str| {
                if (self.anchors.contains(token_str)) {
                    try anchors_out.append(byte_pos);
                }
            }
        }
    }

    fn appendUnknownForSlice(self: *const MGT, slice: []const u8, byte_pos: usize, out_tokens: *std.ArrayList(u32), out_anchors: ?*std.ArrayList(usize)) !void {
        const tid = self.unknownReplacement(slice);
        try self.emitToken(tid, byte_pos, out_tokens, out_anchors);
    }

    fn appendBPEOrUnknown(self: *MGT, slice: []const u8, byte_pos: usize, out_tokens: *std.ArrayList(u32), out_anchors: ?*std.ArrayList(usize)) !void {
        const tokens = try self.encodeBPE(slice);
        defer self.allocator.free(tokens);

        if (tokens.len == 0) {
            try self.appendUnknownForSlice(slice, byte_pos, out_tokens, out_anchors);
            return;
        }

        for (tokens) |tid| {
            try self.emitToken(tid, byte_pos, out_tokens, out_anchors);
        }
    }

    fn encodeInternal(self: *MGT, text: []const u8, out_tokens: *std.ArrayList(u32), out_anchors: ?*std.ArrayList(usize)) !void {
        var i: usize = 0;

        while (i < text.len) {
            if (self.getKnownSpecialTokenLen(text, i)) |special_len| {
                const special_token = text[i .. i + special_len];
                const tid = self.token_to_id.get(special_token) orelse return error.InvalidData;
                try self.emitToken(tid, i, out_tokens, out_anchors);
                i += special_len;
                continue;
            }

            if (isWhitespace(text[i])) {
                const whitespace = text[i .. i + 1];

                if (self.token_to_id.get(whitespace)) |tid| {
                    try self.emitToken(tid, i, out_tokens, out_anchors);
                } else if (text[i] == ' ') {
                    if (self.token_to_id.get(" ")) |space_tid| {
                        try self.emitToken(space_tid, i, out_tokens, out_anchors);
                    } else {
                        try self.appendUnknownForSlice(whitespace, i, out_tokens, out_anchors);
                    }
                } else {
                    try self.appendUnknownForSlice(whitespace, i, out_tokens, out_anchors);
                }

                i += 1;
                continue;
            }

            if (isPunctuation(text[i])) {
                const punctuation = text[i .. i + 1];

                if (self.token_to_id.get(punctuation)) |tid| {
                    try self.emitToken(tid, i, out_tokens, out_anchors);
                } else {
                    try self.appendBPEOrUnknown(punctuation, i, out_tokens, out_anchors);
                }

                i += 1;
                continue;
            }

            var word_end = i;
            while (word_end < text.len) {
                if (self.isKnownSpecialTokenStart(text, word_end)) break;

                const c = text[word_end];
                if (isWhitespace(c) or isPunctuation(c)) break;

                const char_len = safeUtf8SequenceLenAt(text, word_end);
                if (char_len == 0) break;
                word_end += char_len;
            }

            if (word_end == i) {
                const char_len = safeUtf8SequenceLenAt(text, i);
                if (char_len == 0 or char_len > text.len - i) return error.InvalidData;
                try self.appendBPEOrUnknown(text[i .. i + char_len], i, out_tokens, out_anchors);
                i += char_len;
                continue;
            }

            const word = text[i..word_end];

            if (self.token_to_id.get(word)) |tid| {
                try self.emitToken(tid, i, out_tokens, out_anchors);
            } else if (try self.morphDecompose(word, i, out_tokens, out_anchors)) {
            } else {
                try self.subwordSplitInto(word, i, out_tokens, out_anchors);
            }

            i = word_end;
        }
    }

    pub fn encode(self: *MGT, text: []const u8, out_tokens: *std.ArrayList(u32)) !void {
        try self.encodeInternal(text, out_tokens, null);
    }

    pub fn encodeBounded(
        self: *MGT,
        text: []const u8,
        max_tokens: usize,
        out_tokens: *std.ArrayList(u32),
    ) !void {
        if (max_tokens == 0) return;
        const max_bytes = std.math.mul(usize, max_tokens, 4) catch return error.InputTooLarge;
        const bounded = truncateUtf8Input(text, max_bytes);
        const original_len = out_tokens.items.len;
        try self.encodeInternal(bounded, out_tokens, null);
        if (out_tokens.items.len - original_len > max_tokens) {
            out_tokens.shrinkRetainingCapacity(original_len + max_tokens);
        }
    }

    fn binarySearchString(sorted: []const []const u8, target: []const u8) bool {
        var lo: usize = 0;
        var hi: usize = sorted.len;

        while (lo < hi) {
            const mid = lo + (hi - lo) / 2;

            if (std.mem.lessThan(u8, sorted[mid], target)) {
                lo = mid + 1;
            } else if (std.mem.lessThan(u8, target, sorted[mid])) {
                hi = mid;
            } else {
                return true;
            }
        }

        return false;
    }

    fn findLongestPrefix(self: *MGT, word: []const u8) ?struct { prefix: []const u8, len: usize } {
        if (word.len < 2) return null;

        var test_len = @min(self.max_prefix_len, word.len - 1);
        while (test_len > 0) : (test_len -= 1) {
            const candidate = word[0..test_len];

            if (self.prefixes.get(candidate)) |id| {
                const key = self.id_to_token.get(id) orelse return null;
                return .{ .prefix = key, .len = test_len };
            }
        }

        return null;
    }

    fn findLongestSuffix(self: *MGT, word: []const u8) ?struct { suffix: []const u8, len: usize } {
        if (word.len < 2) return null;

        var test_len = @min(self.max_suffix_len, word.len - 1);
        while (test_len > 0) : (test_len -= 1) {
            const candidate = word[word.len - test_len ..];

            if (self.suffixes.get(candidate)) |id| {
                const key = self.id_to_token.get(id) orelse return null;
                return .{ .suffix = key, .len = test_len };
            }
        }

        return null;
    }

    fn morphDecompose(self: *MGT, word: []const u8, word_start: usize, out_tokens: *std.ArrayList(u32), out_anchors: ?*std.ArrayList(usize)) !bool {
        if (word.len < 4) return false;

        const prefix_result = self.findLongestPrefix(word);
        const suffix_result = self.findLongestSuffix(word);

        const prefix_len = if (prefix_result) |prefix| prefix.len else 0;
        const suffix_len = if (suffix_result) |suffix| suffix.len else 0;

        if (prefix_len == 0 and suffix_len == 0) return false;
        if (prefix_len > word.len or suffix_len > word.len - prefix_len) return false;

        const root_start = prefix_len;
        const root_end = word.len - suffix_len;

        if (root_end <= root_start or root_end - root_start < 2) return false;

        const root = word[root_start..root_end];
        const root_tid = self.token_to_id.get(root) orelse return false;

        if (prefix_result) |prefix| {
            const tid = self.token_to_id.get(prefix.prefix) orelse return false;
            try self.emitToken(tid, word_start, out_tokens, out_anchors);
        }

        try self.emitToken(root_tid, word_start + root_start, out_tokens, out_anchors);

        if (suffix_result) |suffix| {
            const tid = self.token_to_id.get(suffix.suffix) orelse return false;
            try self.emitToken(tid, word_start + word.len - suffix.len, out_tokens, out_anchors);
        }

        return true;
    }

    fn addByteToken(self: *MGT, byte: u8) !u32 {
        if (self.byte_token_ids[byte]) |existing| return existing;
        if (!self.allow_dynamic_tokens) return SPECIAL_TOKENS.UNK;

        var buffer: [16]u8 = undefined;
        const byte_string = try std.fmt.bufPrint(&buffer, "<{x:0>2}>", .{byte});
        const id = try self.addToken(byte_string);

        try self.byte_token_values.put(id, byte);
        self.byte_token_ids[byte] = id;

        return id;
    }

    fn addToken(self: *MGT, token: []const u8) !u32 {
        if (self.token_to_id.get(token)) |existing| {
            return existing;
        }

        if (!self.canAddToken()) {
            return error.VocabularyFull;
        }

        const id = self.next_token_id;
        const next_id = std.math.add(u32, id, 1) catch return error.TokenIdOverflow;
        const token_copy = try self.allocator.dupe(u8, token);
        errdefer self.allocator.free(token_copy);

        try self.token_to_id.put(token_copy, id);
        errdefer _ = self.token_to_id.remove(token_copy);

        try self.id_to_token.put(id, token_copy);
        errdefer _ = self.id_to_token.remove(id);

        try self.allocated_strings.append(token_copy);
        self.next_token_id = next_id;

        return id;
    }

    fn adoptTokenWithId(self: *MGT, token: []u8, id: u32) !void {
        if (id == std.math.maxInt(u32)) return error.TokenIdOverflow;
        if (self.token_to_id.contains(token) or self.id_to_token.contains(id)) {
            return error.InvalidData;
        }

        errdefer self.allocator.free(token);

        try self.token_to_id.put(token, id);
        errdefer _ = self.token_to_id.remove(token);

        try self.id_to_token.put(id, token);
        errdefer _ = self.id_to_token.remove(id);

        try self.allocated_strings.append(token);

        if (id >= self.next_token_id) {
            self.next_token_id = std.math.add(u32, id, 1) catch return error.TokenIdOverflow;
        }
    }

    fn getCanonicalTokenForLoad(self: *MGT, raw: []u8, id: u32) ![]const u8 {
        if (self.id_to_token.get(id)) |canonical| {
            if (!mem.eql(u8, canonical, raw)) return error.InvalidData;
            self.allocator.free(raw);
            return canonical;
        }

        if (self.token_to_id.get(raw)) |existing_id| {
            if (existing_id != id) return error.InvalidData;
            const canonical = self.id_to_token.get(existing_id) orelse return error.InvalidData;
            self.allocator.free(raw);
            return canonical;
        }

        try self.adoptTokenWithId(raw, id);
        return self.id_to_token.get(id) orelse return error.InvalidData;
    }

    fn encodeBPE(self: *MGT, text: []const u8) ![]u32 {
        if (text.len == 0) return self.allocator.alloc(u32, 0);

        var current = std.ArrayList(u32).init(self.allocator);
        defer current.deinit();

        for (text) |byte| {
            const tid = if (self.byte_token_ids[byte]) |existing|
                existing
            else
                try self.addByteToken(byte);

            try current.append(tid);
        }

        var pair_cache = std.AutoHashMap(TokenPairKey, BPEMerge).init(self.allocator);
        defer pair_cache.deinit();

        while (current.items.len > 1) {
            var best_priority: u32 = std.math.maxInt(u32);
            var best_left: u32 = 0;
            var best_right: u32 = 0;
            var best_merge_id: u32 = 0;
            var found = false;

            var i: usize = 0;
            while (i < current.items.len - 1) : (i += 1) {
                const left_id = current.items[i];
                const right_id = current.items[i + 1];
                const cache_key = TokenPairKey{ .first = left_id, .second = right_id };

                if (pair_cache.get(cache_key)) |merge| {
                    if (merge.priority < best_priority) {
                        best_priority = merge.priority;
                        best_left = left_id;
                        best_right = right_id;
                        best_merge_id = merge.token_id;
                        found = true;
                    }
                    continue;
                }

                const merge = self.bpe_pairs.get(packPair(left_id, right_id)) orelse BPEMerge{
                    .token_id = 0,
                    .priority = std.math.maxInt(u32),
                };

                try pair_cache.put(cache_key, merge);

                if (merge.priority < best_priority) {
                    best_priority = merge.priority;
                    best_left = left_id;
                    best_right = right_id;
                    best_merge_id = merge.token_id;
                    found = true;
                }
            }

            if (!found) break;

            var write: usize = 0;
            var read: usize = 0;

            while (read < current.items.len) {
                if (read < current.items.len - 1 and
                    current.items[read] == best_left and
                    current.items[read + 1] == best_right)
                {
                    current.items[write] = best_merge_id;
                    write += 1;
                    read += 2;
                } else {
                    if (write != read) {
                        current.items[write] = current.items[read];
                    }
                    write += 1;
                    read += 1;
                }
            }

            current.shrinkRetainingCapacity(write);

            pair_cache.clearRetainingCapacity();
        }

        return current.toOwnedSlice();
    }

    fn validateAllocationCount(comptime T: type, count: usize) !void {
        const bytes = std.math.mul(usize, count, @sizeOf(T)) catch return error.InputTooLarge;
        _ = std.math.add(usize, bytes, 65535) catch return error.InputTooLarge;
    }

    fn validateCorpus(corpus: []const []const u8) !void {
        try validateAllocationCount(BpeSequence, corpus.len);

        var total_tokens: usize = 0;
        for (corpus) |text| {
            try validateAllocationCount(u32, text.len);
            total_tokens = std.math.add(usize, total_tokens, text.len) catch return error.CorpusTooLarge;
        }

        try validateAllocationCount(u32, total_tokens);
    }

    fn bpeScanWorkerFn(ctx: *BpeScanWorkerCtx) void {
        for (ctx.sequences) |*sequence| {
            const seq = sequence.items();
            if (seq.len < 2) continue;

            var i: usize = 0;
            while (i < seq.len - 1) : (i += 1) {
                const key = TokenPairKey{
                    .first = seq[i],
                    .second = seq[i + 1],
                };

                const entry = ctx.pair_freqs.getOrPut(key) catch |err| {
                    ctx.err = err;
                    return;
                };

                if (entry.found_existing) {
                    entry.value_ptr.* = std.math.add(u64, entry.value_ptr.*, 1) catch {
                        ctx.err = error.FrequencyOverflow;
                        return;
                    };
                } else {
                    entry.value_ptr.* = 1;
                }
            }
        }
    }

    fn bpeRebuildWorkerFn(ctx: *BpeRebuildWorkerCtx) void {
        for (ctx.sequences) |*sequence| {
            var seq = sequence.mutableItems();
            if (seq.len < 2) continue;

            var write: usize = 0;
            var read: usize = 0;

            while (read < seq.len) {
                if (read < seq.len - 1 and
                    seq[read] == ctx.best_first and
                    seq[read + 1] == ctx.best_second)
                {
                    seq[write] = ctx.merge_id;
                    write += 1;
                    read += 2;
                } else {
                    if (write != read) {
                        seq[write] = seq[read];
                    }
                    write += 1;
                    read += 1;
                }
            }

            sequence.len = write;
        }
    }

    fn scanBpePairsParallel(sequences: []BpeSequence, requested_workers: usize, transient_allocator: Allocator, destination: *std.AutoHashMap(TokenPairKey, u64)) !void {
        if (sequences.len == 0) return;

        const worker_count = @min(@max(@as(usize, 1), requested_workers), sequences.len);
        const contexts = try transient_allocator.alloc(BpeScanWorkerCtx, worker_count);
        defer transient_allocator.free(contexts);

        const threads = try transient_allocator.alloc(std.Thread, worker_count);
        defer transient_allocator.free(threads);

        const base_chunk = sequences.len / worker_count;
        const remainder = sequences.len % worker_count;

        var offset: usize = 0;
        for (contexts, 0..) |*ctx, worker_index| {
            const chunk = base_chunk + @as(usize, if (worker_index < remainder) 1 else 0);
            ctx.* = .{
                .sequences = sequences[offset .. offset + chunk],
                .pair_freqs = std.AutoHashMap(TokenPairKey, u64).init(transient_allocator),
                .err = null,
            };
            offset += chunk;
        }

        defer {
            for (contexts) |*ctx| {
                ctx.pair_freqs.deinit();
            }
        }

        var spawned: usize = 0;
        while (spawned < worker_count) {
            threads[spawned] = std.Thread.spawn(.{}, bpeScanWorkerFn, .{&contexts[spawned]}) catch |err| {
                for (threads[0..spawned]) |thread| {
                    thread.join();
                }
                return err;
            };
            spawned += 1;
        }

        for (threads[0..spawned]) |thread| {
            thread.join();
        }

        for (contexts) |*ctx| {
            if (ctx.err) |err| return err;
        }

        for (contexts) |*ctx| {
            var iterator = ctx.pair_freqs.iterator();
            while (iterator.next()) |entry| {
                const global_entry = try destination.getOrPut(entry.key_ptr.*);

                if (global_entry.found_existing) {
                    global_entry.value_ptr.* = std.math.add(
                        u64,
                        global_entry.value_ptr.*,
                        entry.value_ptr.*,
                    ) catch return error.FrequencyOverflow;
                } else {
                    global_entry.value_ptr.* = entry.value_ptr.*;
                }
            }
        }
    }

    fn rebuildBpeSequencesParallel(sequences: []BpeSequence, requested_workers: usize, transient_allocator: Allocator, best_key: TokenPairKey, merge_id: u32) !void {
        if (sequences.len == 0) return;

        const worker_count = @min(@max(@as(usize, 1), requested_workers), sequences.len);
        const contexts = try transient_allocator.alloc(BpeRebuildWorkerCtx, worker_count);
        defer transient_allocator.free(contexts);

        const threads = try transient_allocator.alloc(std.Thread, worker_count);
        defer transient_allocator.free(threads);

        const base_chunk = sequences.len / worker_count;
        const remainder = sequences.len % worker_count;

        var offset: usize = 0;
        for (contexts, 0..) |*ctx, worker_index| {
            const chunk = base_chunk + @as(usize, if (worker_index < remainder) 1 else 0);
            ctx.* = .{
                .sequences = sequences[offset .. offset + chunk],
                .best_first = best_key.first,
                .best_second = best_key.second,
                .merge_id = merge_id,
            };
            offset += chunk;
        }

        var spawned: usize = 0;
        while (spawned < worker_count) {
            threads[spawned] = std.Thread.spawn(.{}, bpeRebuildWorkerFn, .{&contexts[spawned]}) catch |err| {
                for (threads[0..spawned]) |thread| {
                    thread.join();
                }
                return err;
            };
            spawned += 1;
        }

        for (threads[0..spawned]) |thread| {
            thread.join();
        }
    }

    fn initialBpePriority(self: *const MGT) !u32 {
        var next_priority: u32 = 0;
        var iterator = self.bpe_pairs.iterator();

        while (iterator.next()) |entry| {
            const priority = entry.value_ptr.priority;
            if (priority >= next_priority) {
                next_priority = std.math.add(u32, priority, 1) catch return error.PriorityOverflow;
            }
        }

        return next_priority;
    }

    pub fn trainBPE(self: *MGT, corpus: []const []const u8, target_vocab_size: u32) !void {
        const configured_target = if (self.max_vocab_size) |limit|
            @min(target_vocab_size, limit)
        else
            target_vocab_size;

        if (self.vocabSize() >= @as(usize, configured_target)) return;

        try validateCorpus(corpus);

        const transient_allocator = std.heap.page_allocator;
        const sequences = try transient_allocator.alloc(BpeSequence, corpus.len);
        var sequence_count: usize = 0;

        defer {
            for (sequences[0..sequence_count]) |sequence| {
                transient_allocator.free(sequence.storage);
            }
            transient_allocator.free(sequences);
        }

        for (corpus) |text| {
            if (text.len == 0) continue;

            try validateAllocationCount(u32, text.len);
            const storage = try transient_allocator.alloc(u32, text.len);
            errdefer transient_allocator.free(storage);

            for (text, 0..) |byte, index| {
                storage[index] = try self.addByteToken(byte);
            }

            sequences[sequence_count] = .{
                .storage = storage,
                .len = storage.len,
            };
            sequence_count += 1;
        }

        if (sequence_count == 0) return;
        if (self.vocabSize() >= @as(usize, configured_target)) return;

        const cpu_count = std.Thread.getCpuCount() catch 1;
        const worker_count = @min(@max(@as(usize, 1), cpu_count), sequence_count);

        var pair_freqs = std.AutoHashMap(TokenPairKey, u64).init(transient_allocator);
        defer pair_freqs.deinit();

        try scanBpePairsParallel(
            sequences[0..sequence_count],
            worker_count,
            transient_allocator,
            &pair_freqs,
        );

        var merge_priority = try self.initialBpePriority();

        while (self.vocabSize() < @as(usize, configured_target)) {
            var best: ?PairFreq = null;
            var iterator = pair_freqs.iterator();

            while (iterator.next()) |entry| {
                const candidate = PairFreq{
                    .key = entry.key_ptr.*,
                    .freq = entry.value_ptr.*,
                };

                if (candidate.freq < 2) continue;

                if (best == null or
                    candidate.freq > best.?.freq or
                    (candidate.freq == best.?.freq and candidate.key.first < best.?.key.first) or
                    (candidate.freq == best.?.freq and
                        candidate.key.first == best.?.key.first and
                        candidate.key.second < best.?.key.second))
                {
                    best = candidate;
                }
            }

            const selected = best orelse break;
            const first_string = self.id_to_token.get(selected.key.first) orelse return error.InvalidData;
            const second_string = self.id_to_token.get(selected.key.second) orelse return error.InvalidData;
            const merged_text = try std.fmt.allocPrint(self.allocator, "{s}{s}", .{ first_string, second_string });
            defer self.allocator.free(merged_text);

            const vocabulary_size_before = self.vocabSize();
            const merge_token_id = try self.addToken(merged_text);
            try self.bpe_pairs.put(packPair(selected.key.first, selected.key.second), .{
                .token_id = merge_token_id,
                .priority = merge_priority,
            });

            try rebuildBpeSequencesParallel(
                sequences[0..sequence_count],
                worker_count,
                transient_allocator,
                selected.key,
                merge_token_id,
            );

            pair_freqs.clearRetainingCapacity();

            try scanBpePairsParallel(
                sequences[0..sequence_count],
                worker_count,
                transient_allocator,
                &pair_freqs,
            );

            if (merge_priority == std.math.maxInt(u32)) {
                if (self.vocabSize() < @as(usize, configured_target)) {
                    return error.PriorityOverflow;
                }
            } else {
                merge_priority += 1;
            }

            if (self.vocabSize() == vocabulary_size_before and pair_freqs.count() == 0) {
                break;
            }
        }
    }

    pub fn decode(self: *MGT, tokens: []const u32, out_text: *std.ArrayList(u8)) !void {
        for (tokens) |token| {
            if (self.byte_token_values.get(token)) |byte| {
                try out_text.append(byte);
            } else if (self.id_to_token.get(token)) |token_string| {
                try out_text.appendSlice(token_string);
            } else {
                const unknown = self.id_to_token.get(SPECIAL_TOKENS.UNK) orelse "[UNK]";
                try out_text.appendSlice(unknown);
            }
        }
    }

    pub fn longestMatch(self: *MGT, text: []const u8, start: usize) usize {
        if (start >= text.len) return 0;

        var max_len: usize = 0;
        var end = start;

        while (end < text.len) {
            const step = safeUtf8SequenceLenAt(text, end);
            if (step == 0 or step > text.len - end) break;

            end += step;
            const substring = text[start..end];

            if (self.token_to_id.contains(substring)) {
                max_len = end - start;
            }
        }

        return max_len;
    }

    pub fn vocabSize(self: *const MGT) usize {
        return self.token_to_id.count();
    }

    pub fn addVocabWord(self: *MGT, word: []const u8, is_anchor: bool) !void {
        if (!self.canAddToken() and !self.token_to_id.contains(word)) {
            return error.VocabularyFull;
        }

        const id = try self.addToken(word);

        if (is_anchor) {
            const key = self.id_to_token.get(id) orelse return error.InvalidData;
            try self.anchors.put(key, @as(u64, id));
        }
    }

    pub fn removeVocabWord(self: *MGT, word: []const u8) void {
        if (mem.eql(u8, word, "[PAD]") or
            mem.eql(u8, word, "[UNK]") or
            mem.eql(u8, word, "[BOS]") or
            mem.eql(u8, word, "[EOS]"))
        {
            return;
        }

        if (self.token_to_id.get(word)) |id| {
            if (self.id_to_token.get(id)) |allocated_pointer| {
                _ = self.token_to_id.remove(word);
                _ = self.id_to_token.remove(id);
                _ = self.anchors.remove(word);
                _ = self.prefixes.remove(word);
                _ = self.suffixes.remove(word);
                _ = self.roots.remove(word);

                if (self.byte_token_values.fetchRemove(id)) |entry| {
                    self.byte_token_ids[entry.value] = null;
                }

                var bpe_remove = std.ArrayList(u64).init(self.allocator);
                defer bpe_remove.deinit();

                var bpe_iterator = self.bpe_pairs.iterator();
                while (bpe_iterator.next()) |entry| {
                    const key = entry.key_ptr.*;
                    const merge = entry.value_ptr.*;

                    if (merge.token_id == id) {
                        bpe_remove.append(key) catch return;
                    }
                }

                for (bpe_remove.items) |key| {
                    _ = self.bpe_pairs.remove(key);
                }

                var index: usize = 0;
                while (index < self.allocated_strings.items.len) : (index += 1) {
                    const str = self.allocated_strings.items[index];

                    if (str.ptr == allocated_pointer.ptr and str.len == allocated_pointer.len) {
                        self.allocator.free(str);
                        _ = self.allocated_strings.swapRemove(index);
                        break;
                    }
                }
            }
        }

        self.rebuildSortedMorphemes() catch {};
    }

    pub fn tokenizeWithAnchors(self: *MGT, text: []const u8, out_tokens: *std.ArrayList(u32), out_anchors: *std.ArrayList(usize)) !void {
        try self.encodeInternal(text, out_tokens, out_anchors);
    }

    pub fn detokenize(self: *MGT, tokens: []const u32) ![]u8 {
        return self.detokenizeAlloc(tokens, self.allocator);
    }

    fn detokenizeAlloc(self: *MGT, tokens: []const u32, allocator: Allocator) ![]u8 {
        var text = std.ArrayList(u8).init(allocator);
        defer text.deinit();

        try self.decode(tokens, &text);
        return text.toOwnedSlice();
    }

    pub fn encodeBatch(self: *MGT, texts: []const []const u8, allocator: Allocator) ![][]u32 {
        const results = try allocator.alloc([]u32, texts.len);
        errdefer allocator.free(results);

        var initialized: usize = 0;
        errdefer {
            for (results[0..initialized]) |result| {
                allocator.free(result);
            }
        }

        for (texts) |text| {
            var tokens = std.ArrayList(u32).init(allocator);
            defer tokens.deinit();

            try self.encode(text, &tokens);
            results[initialized] = try tokens.toOwnedSlice();
            initialized += 1;
        }

        return results;
    }

    pub fn batchDetokenize(self: *MGT, token_lists: []const []const u32, allocator: Allocator) ![][]u8 {
        const results = try allocator.alloc([]u8, token_lists.len);
        errdefer allocator.free(results);

        var initialized: usize = 0;
        errdefer {
            for (results[0..initialized]) |result| {
                allocator.free(result);
            }
        }

        for (token_lists) |tokens| {
            results[initialized] = try self.detokenizeAlloc(tokens, allocator);
            initialized += 1;
        }

        return results;
    }

    fn usizeToU32(value: usize) !u32 {
        if (value > std.math.maxInt(u32)) return error.DataTooLarge;
        return @intCast(value);
    }

    fn writeStringMapSorted(map: std.StringHashMap(u32), writer: anytype, allocator: Allocator) !void {
        const Item = struct {
            key: []const u8,
            value: u32,
        };

        const Context = struct {
            fn lessThan(_: @This(), a: Item, b: Item) bool {
                if (a.value != b.value) return a.value < b.value;
                return std.mem.lessThan(u8, a.key, b.key);
            }
        };

        var items = std.ArrayList(Item).init(allocator);
        defer items.deinit();

        var iterator = map.iterator();
        while (iterator.next()) |entry| {
            try items.append(.{
                .key = entry.key_ptr.*,
                .value = entry.value_ptr.*,
            });
        }

        std.mem.sort(Item, items.items, Context{}, Context.lessThan);
        try writer.writeInt(u32, try usizeToU32(items.items.len), .little);

        for (items.items) |item| {
            try writer.writeInt(u32, try usizeToU32(item.key.len), .little);
            try writer.writeAll(item.key);
            try writer.writeInt(u32, item.value, .little);
        }
    }

    pub fn saveVocab(self: *MGT, path: []const u8) !void {
        var file = try createFilePath(path, .{ .truncate = true });
        defer file.close();

        var writer = file.writer();

        const TokenItem = struct {
            id: u32,
            token: []const u8,
        };

        const TokenContext = struct {
            fn lessThan(_: @This(), a: TokenItem, b: TokenItem) bool {
                if (a.id != b.id) return a.id < b.id;
                return std.mem.lessThan(u8, a.token, b.token);
            }
        };

        var token_items = std.ArrayList(TokenItem).init(self.allocator);
        defer token_items.deinit();

        var token_iterator = self.id_to_token.iterator();
        while (token_iterator.next()) |entry| {
            try token_items.append(.{
                .id = entry.key_ptr.*,
                .token = entry.value_ptr.*,
            });
        }

        std.mem.sort(TokenItem, token_items.items, TokenContext{}, TokenContext.lessThan);
        try writer.writeInt(u32, try usizeToU32(token_items.items.len), .little);

        for (token_items.items) |item| {
            try writer.writeInt(u32, try usizeToU32(item.token.len), .little);
            try writer.writeAll(item.token);
            try writer.writeInt(u32, item.id, .little);
        }

        const BpeItem = struct {
            key: u64,
            merge: BPEMerge,
        };

        const BpeContext = struct {
            fn lessThan(_: @This(), a: BpeItem, b: BpeItem) bool {
                if (a.merge.priority != b.merge.priority) {
                    return a.merge.priority < b.merge.priority;
                }
                return a.key < b.key;
            }
        };

        var bpe_items = std.ArrayList(BpeItem).init(self.allocator);
        defer bpe_items.deinit();

        var bpe_iterator = self.bpe_pairs.iterator();
        while (bpe_iterator.next()) |entry| {
            try bpe_items.append(.{
                .key = entry.key_ptr.*,
                .merge = entry.value_ptr.*,
            });
        }

        std.mem.sort(BpeItem, bpe_items.items, BpeContext{}, BpeContext.lessThan);
        try writer.writeInt(u32, try usizeToU32(bpe_items.items.len), .little);

        for (bpe_items.items) |item| {
            try writer.writeInt(u64, item.key, .little);
            try writer.writeInt(u32, item.merge.token_id, .little);
            try writer.writeInt(u32, item.merge.priority, .little);
        }

        try writeStringMapSorted(self.prefixes, writer, self.allocator);
        try writeStringMapSorted(self.suffixes, writer, self.allocator);
        try writeStringMapSorted(self.roots, writer, self.allocator);

        const AnchorItem = struct {
            key: []const u8,
            value: u64,
        };

        const AnchorContext = struct {
            fn lessThan(_: @This(), a: AnchorItem, b: AnchorItem) bool {
                if (a.value != b.value) return a.value < b.value;
                return std.mem.lessThan(u8, a.key, b.key);
            }
        };

        var anchor_items = std.ArrayList(AnchorItem).init(self.allocator);
        defer anchor_items.deinit();

        var anchor_iterator = self.anchors.iterator();
        while (anchor_iterator.next()) |entry| {
            try anchor_items.append(.{
                .key = entry.key_ptr.*,
                .value = entry.value_ptr.*,
            });
        }

        std.mem.sort(AnchorItem, anchor_items.items, AnchorContext{}, AnchorContext.lessThan);
        try writer.writeInt(u32, try usizeToU32(anchor_items.items.len), .little);

        for (anchor_items.items) |item| {
            try writer.writeInt(u32, try usizeToU32(item.key.len), .little);
            try writer.writeAll(item.key);
            try writer.writeInt(u64, item.value, .little);
        }
    }

    pub fn loadVocab(self: *MGT, path: []const u8) !void {
        self.reset();

        var file = try openFilePath(path, .{});
        defer file.close();

        var reader = file.reader();
        const token_count = try reader.readInt(u32, .little);

        if (self.max_vocab_size) |limit| {
            if (token_count > limit) return error.VocabularyFull;
        }

        var token_index: u32 = 0;
        while (token_index < token_count) : (token_index += 1) {
            const word_len = try reader.readInt(u32, .little);
            const word_buffer = try self.allocator.alloc(u8, @as(usize, word_len));
            var word_owned = true;
            errdefer if (word_owned) self.allocator.free(word_buffer);

            try reader.readNoEof(word_buffer);
            const id = try reader.readInt(u32, .little);

            if (id == std.math.maxInt(u32) or
                self.token_to_id.contains(word_buffer) or
                self.id_to_token.contains(id))
            {
                return error.InvalidData;
            }

            try self.adoptTokenWithId(word_buffer, id);
            word_owned = false;
        }

        const bpe_count = try reader.readInt(u32, .little);
        var bpe_index: u32 = 0;

        while (bpe_index < bpe_count) : (bpe_index += 1) {
            const pair_key = try reader.readInt(u64, .little);
            const token_id = try reader.readInt(u32, .little);
            const priority = try reader.readInt(u32, .little);

            try self.bpe_pairs.put(pair_key, .{
                .token_id = token_id,
                .priority = priority,
            });
        }

        const ReadStringMap = struct {
            fn read(self_mgt: *MGT, map: *std.StringHashMap(u32), input_reader: anytype) !void {
                const count = try input_reader.readInt(u32, .little);
                var index: u32 = 0;

                while (index < count) : (index += 1) {
                    const length = try input_reader.readInt(u32, .little);
                    const buffer = try self_mgt.allocator.alloc(u8, @as(usize, length));
                    var buffer_owned = true;
                    errdefer if (buffer_owned) self_mgt.allocator.free(buffer);

                    try input_reader.readNoEof(buffer);

                    const id = try input_reader.readInt(u32, .little);
                    const canonical = try self_mgt.getCanonicalTokenForLoad(buffer, id);
                    buffer_owned = false;

                    try map.put(canonical, id);
                }
            }
        };

        try ReadStringMap.read(self, &self.prefixes, reader);
        try ReadStringMap.read(self, &self.suffixes, reader);
        try ReadStringMap.read(self, &self.roots, reader);

        const anchor_count = try reader.readInt(u32, .little);
        var anchor_index: u32 = 0;

        while (anchor_index < anchor_count) : (anchor_index += 1) {
            const key_len = try reader.readInt(u32, .little);
            const key_buffer = try self.allocator.alloc(u8, @as(usize, key_len));
            var key_owned = true;
            errdefer if (key_owned) self.allocator.free(key_buffer);

            try reader.readNoEof(key_buffer);

            const value = try reader.readInt(u64, .little);
            if (value > std.math.maxInt(u32)) return error.InvalidData;

            const canonical = try self.getCanonicalTokenForLoad(key_buffer, @intCast(value));
            key_owned = false;

            try self.anchors.put(canonical, value);
        }

        try self.rebuildSortedMorphemes();
        try self.rebuildByteTokenLookup();
        self.allow_dynamic_tokens = false;
    }

    fn rebuildByteTokenLookup(self: *MGT) !void {
        self.byte_token_ids = [_]?u32{null} ** 256;
        self.byte_token_values.clearRetainingCapacity();

        var iterator = self.id_to_token.iterator();
        while (iterator.next()) |entry| {
            const token_string = entry.value_ptr.*;
            const id = entry.key_ptr.*;

            if (token_string.len == 4 and
                token_string[0] == '<' and
                token_string[3] == '>')
            {
                const hex = token_string[1..3];

                if (std.fmt.parseInt(u8, hex, 16)) |byte_value| {
                    if (self.byte_token_ids[byte_value] != null) {
                        return error.InvalidData;
                    }

                    self.byte_token_ids[byte_value] = id;
                    try self.byte_token_values.put(id, byte_value);
                } else |_| {}
            }
        }
    }

    pub fn unknownReplacement(self: *const MGT, context: []const u8) u32 {
        _ = self;
        _ = context;
        return SPECIAL_TOKENS.UNK;
    }

    fn subwordSplitInto(self: *MGT, word: []const u8, word_start: usize, out_tokens: *std.ArrayList(u32), out_anchors: ?*std.ArrayList(usize)) !void {
        var i: usize = 0;

        while (i < word.len) {
            const match_len = self.longestMatch(word, i);

            if (match_len > 0 and match_len <= word.len - i) {
                const found_word = word[i .. i + match_len];

                if (self.token_to_id.get(found_word)) |tid| {
                    try self.emitToken(tid, word_start + i, out_tokens, out_anchors);
                    i += match_len;
                    continue;
                }
            }

            const char_len = safeUtf8SequenceLenAt(word, i);
            if (char_len == 0 or char_len > word.len - i) return error.InvalidData;

            const piece = word[i .. i + char_len];
            try self.appendBPEOrUnknown(piece, word_start + i, out_tokens, out_anchors);
            i += char_len;
        }
    }

    pub fn subwordSplit(self: *MGT, word: []const u8) ![]u32 {
        var tokens = std.ArrayList(u32).init(self.allocator);
        defer tokens.deinit();

        try self.subwordSplitInto(word, 0, &tokens, null);
        return tokens.toOwnedSlice();
    }

    pub fn mergeSubwords(self: *MGT, subwords: []const []const u32) ![]u32 {
        var merged = std.ArrayList(u32).init(self.allocator);
        defer merged.deinit();

        var total_length: usize = 0;
        for (subwords) |subword| {
            total_length = std.math.add(usize, total_length, subword.len) catch return error.InputTooLarge;
        }

        try merged.ensureTotalCapacity(total_length);

        for (subwords) |subword| {
            try merged.appendSlice(subword);
        }

        return merged.toOwnedSlice();
    }

    pub fn validateTokens(self: *MGT, tokens: []const u32) bool {
        for (tokens) |token| {
            if (!self.id_to_token.contains(token)) return false;
        }

        return true;
    }

    pub fn coverage(self: *MGT, corpus: []const u8) f32 {
        if (corpus.len == 0) return 0.0;

        var covered: usize = 0;
        var i: usize = 0;

        while (i < corpus.len) {
            if (self.getKnownSpecialTokenLen(corpus, i)) |special_len| {
                covered = std.math.add(usize, covered, special_len) catch return 0.0;
                i += special_len;
                continue;
            }

            if (isWhitespace(corpus[i]) or isPunctuation(corpus[i])) {
                const slice = corpus[i .. i + 1];

                if (self.token_to_id.contains(slice) or
                    (corpus[i] == ' ' and self.token_to_id.contains(" ")))
                {
                    covered = std.math.add(usize, covered, 1) catch return 0.0;
                }

                i += 1;
                continue;
            }

            var word_end = i;
            while (word_end < corpus.len) {
                if (self.isKnownSpecialTokenStart(corpus, word_end)) break;

                const c = corpus[word_end];
                if (isWhitespace(c) or isPunctuation(c)) break;

                const char_len = safeUtf8SequenceLenAt(corpus, word_end);
                if (char_len == 0 or char_len > corpus.len - word_end) break;
                word_end += char_len;
            }

            if (word_end == i) {
                const char_len = safeUtf8SequenceLenAt(corpus, i);
                if (char_len == 0 or char_len > corpus.len - i) break;

                const maybe_bpe = self.encodeBPE(corpus[i .. i + char_len]) catch null;
                if (maybe_bpe) |bpe| {
                    defer self.allocator.free(bpe);

                    var all_unknown = true;
                    for (bpe) |tid| {
                        if (tid != SPECIAL_TOKENS.UNK) {
                            all_unknown = false;
                            break;
                        }
                    }

                    if (!all_unknown) {
                        covered = std.math.add(usize, covered, char_len) catch return 0.0;
                    }
                }

                i += char_len;
                continue;
            }

            const word = corpus[i..word_end];

            if (self.token_to_id.contains(word)) {
                covered = std.math.add(usize, covered, word.len) catch return 0.0;
            } else {
                var temporary = std.ArrayList(u32).init(self.allocator);
                defer temporary.deinit();

                if (self.morphDecompose(word, i, &temporary, null) catch false) {
                    covered = std.math.add(usize, covered, word.len) catch return 0.0;
                } else {
                    const maybe_subwords = self.subwordSplit(word) catch null;

                    if (maybe_subwords) |subwords| {
                        defer self.allocator.free(subwords);

                        var all_unknown = true;
                        for (subwords) |tid| {
                            if (tid != SPECIAL_TOKENS.UNK) {
                                all_unknown = false;
                                break;
                            }
                        }

                        if (!all_unknown) {
                            covered = std.math.add(usize, covered, word.len) catch return 0.0;
                        }
                    }
                }
            }

            i = word_end;
        }

        return @as(f32, @floatFromInt(covered)) /
            @as(f32, @floatFromInt(corpus.len));
    }

    pub fn encodeToTensor(self: *MGT, text: []const u8, allocator: Allocator) !Tensor {
        var tokens = std.ArrayList(u32).init(allocator);
        defer tokens.deinit();

        try self.encode(text, &tokens);

        const shape = [_]usize{tokens.items.len};
        var tensor = try Tensor.init(allocator, &shape);

        for (tokens.items, 0..) |token, index| {
            tensor.data[index] = @floatFromInt(token);
        }

        return tensor;
    }

    pub fn encodeBatchToTensor(self: *MGT, texts: []const []const u8, allocator: Allocator) !Tensor {
        var max_len: usize = 0;
        var per_row = std.ArrayList([]u32).init(allocator);

        defer {
            for (per_row.items) |row| {
                allocator.free(row);
            }
            per_row.deinit();
        }

        for (texts) |text| {
            var tokens = std.ArrayList(u32).init(allocator);
            defer tokens.deinit();

            try self.encode(text, &tokens);

            const owned = try tokens.toOwnedSlice();
            errdefer allocator.free(owned);

            try per_row.append(owned);
            max_len = @max(max_len, owned.len);
        }

        if (max_len == 0) max_len = 1;
        _ = std.math.mul(usize, texts.len, max_len) catch return error.InputTooLarge;

        const shape = [_]usize{ texts.len, max_len };
        var tensor = try Tensor.init(allocator, &shape);
        @memset(tensor.data, @as(@TypeOf(tensor.data[0]), 0));

        for (per_row.items, 0..) |row, row_index| {
            for (row, 0..) |token, column_index| {
                const row_offset = std.math.mul(usize, row_index, max_len) catch return error.InputTooLarge;
                const tensor_index = std.math.add(usize, row_offset, column_index) catch return error.InputTooLarge;
                tensor.data[tensor_index] = @floatFromInt(token);
            }
        }

        return tensor;
    }

    pub fn decodeFromTensor(self: *MGT, tensor: *const Tensor, allocator: Allocator) ![]u8 {
        const tokens = try allocator.alloc(u32, tensor.data.len);
        defer allocator.free(tokens);

        for (tensor.data, 0..) |value, index| {
            if (std.math.isNan(value) or
                std.math.isInf(value) or
                value < 0.0 or
                value > @as(@TypeOf(value), @floatFromInt(std.math.maxInt(u32))))
            {
                tokens[index] = SPECIAL_TOKENS.UNK;
            } else {
                tokens[index] = @intFromFloat(value);

                if (!self.id_to_token.contains(tokens[index])) {
                    tokens[index] = SPECIAL_TOKENS.UNK;
                }
            }
        }

        return self.detokenizeAlloc(tokens, allocator);
    }
};

test "MGT encode decode" {
    const testing = std.testing;
    const gpa = testing.allocator;
    const vocab = &.{ "hello", "world", " " };
    const anchors = &.{"hello"};

    var mgt = try MGT.init(gpa, vocab, anchors, null, .english);
    defer mgt.deinit();

    var tokens = std.ArrayList(u32).init(gpa);
    defer tokens.deinit();

    try mgt.encode("hello world", &tokens);
    try testing.expect(tokens.items.len >= 3);

    var text = std.ArrayList(u8).init(gpa);
    defer text.deinit();

    try mgt.decode(tokens.items, &text);
    try testing.expectEqualStrings("hello world", text.items);
}

test "MGT add remove vocab" {
    const testing = std.testing;
    const gpa = testing.allocator;

    var mgt = try MGT.init(gpa, &.{}, &.{}, null, .english);
    defer mgt.deinit();

    try mgt.addVocabWord("test", true);
    try testing.expect(mgt.anchors.contains("test"));

    mgt.removeVocabWord("test");

    try testing.expect(!mgt.anchors.contains("test"));
    try testing.expect(!mgt.token_to_id.contains("test"));
}

test "MGT longest match" {
    const testing = std.testing;
    const gpa = testing.allocator;

    var mgt = try MGT.init(gpa, &.{ "hello", "hell" }, &.{}, null, .english);
    defer mgt.deinit();

    const len = mgt.longestMatch("hello", 0);
    try testing.expectEqual(@as(usize, 5), len);
}

test "MGT batch encode" {
    const testing = std.testing;
    const gpa = testing.allocator;

    var mgt = try MGT.init(gpa, &.{ "a", "b" }, &.{}, null, .english);
    defer mgt.deinit();

    const texts = &.{ "a", "b" };
    const batches = try mgt.encodeBatch(texts, gpa);

    defer {
        for (batches) |batch| {
            gpa.free(batch);
        }
        gpa.free(batches);
    }

    try testing.expectEqual(@as(usize, 2), batches.len);
    try testing.expectEqual(@as(usize, 1), batches[0].len);
    try testing.expectEqual(@as(usize, 1), batches[1].len);
}

test "MGT subword split" {
    const testing = std.testing;
    const gpa = testing.allocator;

    var mgt = try MGT.init(gpa, &.{ "hel", "lo" }, &.{}, null, .english);
    defer mgt.deinit();

    const subwords = try mgt.subwordSplit("hello");
    defer gpa.free(subwords);

    try testing.expectEqual(@as(usize, 2), subwords.len);
    try testing.expect(mgt.validateTokens(subwords));
}

test "MGT coverage" {
    const testing = std.testing;
    const gpa = testing.allocator;

    var mgt = try MGT.init(gpa, &.{ "hello", "world", " " }, &.{}, null, .english);
    defer mgt.deinit();

    const result = mgt.coverage("hello world");
    try testing.expect(result > 0.99);
}

test "MGT BPE training" {
    const testing = std.testing;
    const gpa = testing.allocator;
    const corpus = &.{
        "lower",
        "lowest",
        "newer",
        "wider",
        "lower",
        "lowest",
    };

    var mgt = try MGT.init(gpa, &.{}, &.{}, 512, .english);
    defer mgt.deinit();

    try mgt.trainBPE(corpus, 320);

    var encoded = std.ArrayList(u32).init(gpa);
    defer encoded.deinit();

    try mgt.encode("lower", &encoded);
    try testing.expect(encoded.items.len > 0);
    try testing.expect(mgt.validateTokens(encoded.items));

    var decoded = std.ArrayList(u8).init(gpa);
    defer decoded.deinit();

    try mgt.decode(encoded.items, &decoded);
    try testing.expectEqualStrings("lower", decoded.items);
}

test "MGT empty BPE corpus" {
    const testing = std.testing;
    const gpa = testing.allocator;

    var mgt = try MGT.init(gpa, &.{}, &.{}, 512, .english);
    defer mgt.deinit();

    try mgt.trainBPE(&.{}, 320);
    try testing.expect(mgt.vocabSize() >= 4);
}

test "MGT BPE target vocabulary size" {
    const testing = std.testing;
    const gpa = testing.allocator;
    const corpus = &.{
        "aaaaaaaa",
        "aaaaaaaa",
        "abababab",
        "abababab",
    };

    var mgt = try MGT.init(gpa, &.{}, &.{}, 512, .english);
    defer mgt.deinit();

    try mgt.trainBPE(corpus, 300);
    try testing.expect(mgt.vocabSize() <= 300);
}


pub const SSI = struct {
    root: ?*BucketNode,
    allocator: Allocator,
    height: usize = 0,
    size: usize = 0,
    max_height: usize = 6,

    const bucket_width: usize = 6;
    const bucket_count: usize = 1 << bucket_width;
    const tensor_width: usize = 134;

    const Segment = struct {
        tokens: []u32,
        position: u64,
        score: f32,
        anchor_hash: u64,
        signature: u64,

        pub fn init(allocator: Allocator, tokens: []const u32, position: u64, score: f32, anchor_hash: u64) !Segment {
            return .{
                .tokens = try allocator.dupe(u32, tokens),
                .position = position,
                .score = score,
                .anchor_hash = anchor_hash,
                .signature = computeMinHashSignature(tokens),
            };
        }

        pub fn deinit(self: *Segment, allocator: Allocator) void {
            allocator.free(self.tokens);
            self.tokens = &.{};
        }

        pub fn tokenHash(self: *const Segment) u64 {
            return hashTokens(self.tokens);
        }

        pub fn fullHash(self: *const Segment) u64 {
            var state: u64 = 0;
            state = mixHash(state, self.position);
            state = mixHash(state, @as(u64, scoreBits(self.score)));
            state = mixHash(state, self.anchor_hash);
            state = mixHash(state, self.signature);
            state = mixHash(state, @as(u64, @intCast(self.tokens.len)));
            for (self.tokens) |tok| {
                state = mixHash(state, tok);
            }
            return state;
        }
    };

    const CollisionNode = struct {
        seg: Segment,
        next: ?*CollisionNode,
    };

    const BucketNode = struct {
        hash: u64,
        children: ?[]?*BucketNode,
        segment: ?Segment,
        collision_chain: ?*CollisionNode,
        height: usize,
        is_leaf: bool,

        pub fn init(allocator: Allocator, height: usize) !BucketNode {
            var children: ?[]?*BucketNode = null;
            if (height > 0) {
                const allocated = try allocator.alloc(?*BucketNode, bucket_count);
                @memset(allocated, null);
                children = allocated;
            }
            return .{
                .hash = 0,
                .children = children,
                .segment = null,
                .collision_chain = null,
                .height = height,
                .is_leaf = height == 0,
            };
        }

        pub fn deinit(self: *BucketNode, allocator: Allocator) void {
            if (self.segment) |*seg| {
                seg.deinit(allocator);
                self.segment = null;
            }
            var chain = self.collision_chain;
            while (chain) |c| {
                const next = c.next;
                c.seg.deinit(allocator);
                allocator.destroy(c);
                chain = next;
            }
            self.collision_chain = null;
            if (self.children) |children| {
                allocator.free(children);
                self.children = null;
            }
        }
    };

    pub fn init(allocator: Allocator) SSI {
        return .{
            .root = null,
            .allocator = allocator,
            .height = 0,
            .size = 0,
            .max_height = bucket_width,
        };
    }

    fn mixHash(state: u64, value: u64) u64 {
        return state *% 0x9E3779B185EBCA87 +% value +% 0x517CC1B727220A95;
    }

    fn scoreBits(value: f32) u32 {
        return @as(u32, @bitCast(value));
    }

    fn hashTokens(tokens: []const u32) u64 {
        var state: u64 = 0;
        state = mixHash(state, @as(u64, @intCast(tokens.len)));
        for (tokens) |tok| {
            state = mixHash(state, tok);
        }
        return state;
    }

    fn minHashSeedA(lane: u64) u64 {
        return 0x9E3779B185EBCA87 +% lane *% 0xC2B2AE3D27D4EB4F +% 0x165667B19E3779F9;
    }

    fn minHashSeedB(lane: u64) u64 {
        return 0x517CC1B727220A95 +% lane *% 0xD6E8FEB86659FD93 +% 0x2545F4914F6CDD1D;
    }

    fn minHashLaneHash(token: u32, lane: u64) u64 {
        var h = @as(u64, token) *% minHashSeedA(lane) +% minHashSeedB(lane);
        h = (h ^ (h >> 30)) *% 0xbf58476d1ce4e5b9;
        h = (h ^ (h >> 27)) *% 0x94d049bb133111eb;
        h = h ^ (h >> 31);
        return h;
    }

    pub fn computeMinHashSignature(tokens: []const u32) u64 {
        const vector_len: usize = 8;
        const lane_count: usize = 64 / vector_len;
        var minima: [lane_count]@Vector(vector_len, u64) = undefined;
        var lane: usize = 0;
        while (lane < lane_count) : (lane += 1) {
            minima[lane] = @splat(std.math.maxInt(u64));
        }
        for (tokens) |token| {
            lane = 0;
            while (lane < lane_count) : (lane += 1) {
                var hashes: @Vector(vector_len, u64) = undefined;
                var lane_bit: usize = 0;
                while (lane_bit < vector_len) : (lane_bit += 1) {
                    hashes[lane_bit] = minHashLaneHash(token, lane * vector_len + lane_bit);
                }
                minima[lane] = @min(minima[lane], hashes);
            }
        }
        var signature: u64 = 0;
        lane = 0;
        while (lane < lane_count) : (lane += 1) {
            const parities = minima[lane] & @as(@Vector(vector_len, u64), @splat(1));
            var lane_bit: usize = 0;
            while (lane_bit < vector_len) : (lane_bit += 1) {
                if (parities[lane_bit] != 0) {
                    signature |= @as(u64, 1) << @intCast(lane * vector_len + lane_bit);
                }
            }
        }
        return signature;
    }

    pub fn signatureSimilarity(query_signature: u64, segment_signature: u64) f32 {
        const mismatch = @popCount(query_signature ^ segment_signature);
        const matches = 64 - @as(i64, @intCast(mismatch));
        if (matches <= 0) return 0.0;
        const ratio = @as(f32, @floatFromInt(matches)) / 64.0;
        const estimate = 2.0 * ratio - 1.0;
        return std.math.clamp(estimate, @as(f32, 0.0), @as(f32, 1.0));
    }

    fn computeAnchorHash(tokens: []const u32, position: u64) u64 {
        var state: u64 = position;
        state = mixHash(state, @as(u64, @intCast(tokens.len)));
        for (tokens) |tok| {
            state = mixHash(state, tok);
        }
        return state;
    }

    fn bucketIndex(position: u64) usize {
        var h = position *% 0x9E3779B185EBCA87;
        h = (h ^ (h >> 30)) *% 0xbf58476d1ce4e5b9;
        h = (h ^ (h >> 27)) *% 0x94d049bb133111eb;
        h = h ^ (h >> 31);
        return @as(usize, @intCast(h & (bucket_count - 1)));
    }

    fn low32(value: u64) u32 {
        return @as(u32, @intCast(value & 0xFFFF_FFFF));
    }

    fn high32(value: u64) u32 {
        return @as(u32, @intCast(value >> 32));
    }

    fn joinU64(lo: u32, hi: u32) u64 {
        return (@as(u64, hi) << 32) | @as(u64, lo);
    }

    fn bitsToFloat(bits: u32) f32 {
        return @as(f32, @bitCast(bits));
    }

    fn floatToBits(value: f32) u32 {
        return @as(u32, @bitCast(value));
    }

    fn safeFloat(v: f32) f32 {
        if (std.math.isNan(v) or std.math.isInf(v)) return 0.0;
        return std.math.clamp(v, -3.4e38, 3.4e38);
    }

    fn recursiveDeinit(node: *BucketNode, allocator: Allocator) void {
        if (node.children) |children| {
            for (children) |maybe_child| {
                if (maybe_child) |child| {
                    recursiveDeinit(child, allocator);
                }
            }
        }
        node.deinit(allocator);
        allocator.destroy(node);
    }

    pub fn deinit(self: *SSI) void {
        if (self.root) |root| {
            recursiveDeinit(root, self.allocator);
        }
        self.root = null;
        self.height = 0;
        self.size = 0;
    }

    fn computeLeafHash(node: *const BucketNode) u64 {
        var acc: u64 = 0;
        if (node.segment) |seg| {
            acc +%= seg.fullHash();
        }
        var chain = node.collision_chain;
        while (chain) |c| {
            acc +%= c.seg.fullHash();
            chain = c.next;
        }
        return acc;
    }

    fn computeBranchHash(node: *const BucketNode) u64 {
        var acc: u64 = 0;
        if (node.children) |children| {
            for (children) |maybe_child| {
                if (maybe_child) |child| {
                    acc +%= child.hash;
                }
            }
        }
        return acc;
    }

    fn refreshHash(node: *BucketNode) void {
        node.hash = if (node.is_leaf) computeLeafHash(node) else computeBranchHash(node);
    }

    fn ensureRoot(self: *SSI) !*BucketNode {
        if (self.root == null) {
            const root = try self.allocator.create(BucketNode);
            root.* = try BucketNode.init(self.allocator, bucket_width);
            root.is_leaf = false;
            root.height = bucket_width;
            refreshHash(root);
            self.root = root;
            self.height = bucket_width;
        }
        return self.root.?;
    }

    fn insertIntoLeaf(self: *SSI, leaf: *BucketNode, tokens: []const u32, position: u64, score: f32, anchor_hash: u64) !bool {
        if (!leaf.is_leaf or leaf.height != 0) {
            return error.InvalidNodeState;
        }
        if (leaf.segment == null) {
            leaf.segment = try Segment.init(self.allocator, tokens, position, score, anchor_hash);
            refreshHash(leaf);
            return true;
        }
        if (leaf.segment.?.position == position) {
            var old = leaf.segment.?;
            old.deinit(self.allocator);
            leaf.segment = try Segment.init(self.allocator, tokens, position, score, anchor_hash);
            refreshHash(leaf);
            return false;
        }
        var chain = leaf.collision_chain;
        while (chain) |c| {
            if (c.seg.position == position) {
                c.seg.deinit(self.allocator);
                c.seg = try Segment.init(self.allocator, tokens, position, score, anchor_hash);
                refreshHash(leaf);
                return false;
            }
            chain = c.next;
        }
        const collision = try self.allocator.create(CollisionNode);
        collision.* = .{
            .seg = try Segment.init(self.allocator, tokens, position, score, anchor_hash),
            .next = leaf.collision_chain,
        };
        leaf.collision_chain = collision;
        refreshHash(leaf);
        return true;
    }

    fn addSequenceWithMetadata(self: *SSI, tokens: []const u32, position: u64, score: f32, anchor_hash: u64) !void {
        const root = try self.ensureRoot();
        const idx = bucketIndex(position);
        if (root.children.?[idx] == null) {
            const leaf = try self.allocator.create(BucketNode);
            leaf.* = try BucketNode.init(self.allocator, 0);
            root.children.?[idx] = leaf;
        }
        const leaf = root.children.?[idx].?;
        const inserted_new = try self.insertIntoLeaf(leaf, tokens, position, score, anchor_hash);
        refreshHash(root);
        if (inserted_new) {
            self.size += 1;
        }
    }

    fn copyInto(self: *const SSI, target: *SSI) !void {
        if (self.root == null) {
            return;
        }
        const root = self.root.?;
        if (root.children) |children| {
            for (children) |maybe_child| {
                if (maybe_child) |leaf| {
                    if (leaf.segment) |seg| {
                        try target.addSequenceWithMetadata(seg.tokens, seg.position, seg.score, seg.anchor_hash);
                    }
                    var chain = leaf.collision_chain;
                    while (chain) |c| {
                        try target.addSequenceWithMetadata(c.seg.tokens, c.seg.position, c.seg.score, c.seg.anchor_hash);
                        chain = c.next;
                    }
                }
            }
        }
    }

    pub fn addSequence(self: *SSI, tokens: []const u32, position: u64, is_anchor: bool) !void {
        const anchor_hash = if (is_anchor) computeAnchorHash(tokens, position) else 0;
        try self.addSequenceWithMetadata(tokens, position, 0.0, anchor_hash);
        if (self.size > 0) {
            const load_factor = @as(f64, @floatFromInt(self.size)) / @as(f64, @floatFromInt(bucket_count));
            if (load_factor > 8.0) {
                try self.compact();
            }
        }
    }

    pub fn retrieveTopK(self: *const SSI, query_tokens: []const u32, k: usize, allocator: Allocator) ![]RankedSegment {
        if (k == 0) {
            return allocator.alloc(RankedSegment, 0);
        }
        var heap = std.PriorityQueue(RankedSegment, void, struct {
            pub fn lessThan(_: void, a: RankedSegment, b: RankedSegment) std.math.Order {
                return std.math.order(a.score, b.score);
            }
        }.lessThan).init(allocator, {});
        errdefer {
            while (heap.removeOrNull()) |item| {
                var mut = item;
                mut.deinit(allocator);
            }
            heap.deinit();
        }
        defer heap.deinit();
        const query_hash = hashTokens(query_tokens);
        const query_signature = computeMinHashSignature(query_tokens);
        try self.traverse(self.root, query_hash, query_signature, &heap, k, allocator);
        const result_len = @min(k, heap.count());
        var top_n = try allocator.alloc(RankedSegment, result_len);
        var index = result_len;
        while (heap.removeOrNull()) |item| {
            index -= 1;
            top_n[index] = item;
        }
        return top_n;
    }

    fn traverse(self: *const SSI, node: ?*BucketNode, query_hash: u64, query_signature: u64, heap: anytype, k: usize, allocator: Allocator) !void {
        if (node == null) {
            return;
        }
        const current = node.?;
        if (current.is_leaf) {
            if (current.segment) |seg| {
                try addSegmentToHeap(seg, query_hash, query_signature, heap, k, allocator);
            }
            var chain = current.collision_chain;
            while (chain) |c| {
                try addSegmentToHeap(c.seg, query_hash, query_signature, heap, k, allocator);
                chain = c.next;
            }
            return;
        }
        if (current.children) |children| {
            for (children) |maybe_child| {
                if (maybe_child) |child| {
                    try traverse(self, child, query_hash, query_signature, heap, k, allocator);
                }
            }
        }
    }

    fn addSegmentToHeap(seg: Segment, query_hash: u64, query_signature: u64, heap: anytype, k: usize, allocator: Allocator) !void {
        const similarity = computeFusedSimilarity(query_hash, query_signature, seg.tokenHash(), seg.signature);
        if (heap.count() >= k) {
            if (heap.peek()) |top| {
                if (similarity <= top.score) {
                    return;
                }
            }
        }
        const ranked = RankedSegment{
            .tokens = try allocator.dupe(u32, seg.tokens),
            .score = similarity,
            .position = seg.position,
            .anchor = seg.anchor_hash != 0,
        };
        errdefer allocator.free(ranked.tokens);
        if (heap.count() < k) {
            try heap.add(ranked);
            return;
        }
        try heap.add(ranked);
        var removed = heap.remove();
        removed.deinit(allocator);
    }

    fn computeSimilarity(h1: u64, h2: u64) f32 {
        const pc1 = @popCount(h1);
        const pc2 = @popCount(h2);
        if (pc1 == 0 and pc2 == 0) return 1.0;
        if (pc1 == 0 or pc2 == 0) return 0.0;
        const intersection = @popCount(h1 & h2);
        const denom = @sqrt(@as(f32, @floatFromInt(pc1)) * @as(f32, @floatFromInt(pc2)));
        return @as(f32, @floatFromInt(intersection)) / denom;
    }

    fn computeFusedSimilarity(query_hash: u64, query_signature: u64, segment_hash: u64, segment_signature: u64) f32 {
        const hash_cosine = computeSimilarity(query_hash, segment_hash);
        const jaccard_estimate = signatureSimilarity(query_signature, segment_signature);
        return 0.5 * hash_cosine + 0.5 * jaccard_estimate;
    }

    pub fn compact(self: *SSI) !void {
        if (self.size < 1000) {
            return;
        }
        var rebuilt = SSI.init(self.allocator);
        rebuilt.max_height = self.max_height;
        errdefer rebuilt.deinit();
        try self.copyInto(&rebuilt);
        self.deinit();
        self.* = rebuilt;
    }

    pub fn updateScore(self: *SSI, position: u64, new_score: f32) !void {
        const root = self.root orelse return Error.OutOfBounds;
        const child = root.children.?[bucketIndex(position)] orelse return Error.OutOfBounds;
        if (child.segment) |*seg| {
            if (seg.position == position) {
                seg.score = new_score;
                refreshHash(child);
                refreshHash(root);
                return;
            }
        }
        var chain = child.collision_chain;
        while (chain) |c| {
            if (c.seg.position == position) {
                c.seg.score = new_score;
                refreshHash(child);
                refreshHash(root);
                return;
            }
            chain = c.next;
        }
        return Error.OutOfBounds;
    }

    pub fn getSegment(self: *const SSI, position: u64) ?Segment {
        const root = self.root orelse return null;
        const child = root.children.?[bucketIndex(position)] orelse return null;
        if (child.segment) |seg| {
            if (seg.position == position) {
                return seg;
            }
        }
        var chain = child.collision_chain;
        while (chain) |c| {
            if (c.seg.position == position) {
                return c.seg;
            }
            chain = c.next;
        }
        return null;
    }

    fn countSegments(self: *const SSI) usize {
        const root = self.root orelse return 0;
        var count: usize = 0;
        if (root.children) |children| {
            for (children) |maybe_child| {
                if (maybe_child) |leaf| {
                    if (leaf.segment != null) {
                        count += 1;
                    }
                    var chain = leaf.collision_chain;
                    while (chain) |c| {
                        count += 1;
                        chain = c.next;
                    }
                }
            }
        }
        return count;
    }

    fn writeBoolFlag(writer: anytype, value: bool) !void {
        try writer.writeInt(u8, if (value) 1 else 0, .little);
    }

    fn readBoolFlag(reader: anytype) !bool {
        return (try reader.readInt(u8, .little)) != 0;
    }

    fn writeSegment(writer: anytype, seg: Segment) !void {
        try writer.writeInt(u64, seg.position, .little);
        try writer.writeInt(u32, floatToBits(seg.score), .little);
        try writer.writeInt(u64, seg.anchor_hash, .little);
        try writer.writeInt(u64, seg.signature, .little);
        try writer.writeInt(u64, @as(u64, seg.tokens.len), .little);
        for (seg.tokens) |tok| {
            try writer.writeInt(u32, tok, .little);
        }
    }

    fn readSegment(allocator: Allocator, reader: anytype) !Segment {
        const position = try reader.readInt(u64, .little);
        const score = bitsToFloat(try reader.readInt(u32, .little));
        const anchor_hash = try reader.readInt(u64, .little);
        const stored_signature = try reader.readInt(u64, .little);
        const token_len_raw = try reader.readInt(u64, .little);
        if (token_len_raw > std.math.maxInt(usize)) return error.InvalidData;
        const token_len: usize = @intCast(token_len_raw);
        const tokens = try allocator.alloc(u32, token_len);
        errdefer allocator.free(tokens);
        for (tokens) |*tok| {
            tok.* = try reader.readInt(u32, .little);
        }
        const signature = computeMinHashSignature(tokens);
        if (signature != stored_signature) return error.InvalidData;
        return .{
            .tokens = tokens,
            .position = position,
            .score = score,
            .anchor_hash = anchor_hash,
            .signature = signature,
        };
    }

    fn serializeNode(node: *const BucketNode, writer: anytype) !void {
        try writeBoolFlag(writer, node.is_leaf);
        try writer.writeInt(u64, @as(u64, node.height), .little);
        try writer.writeInt(u64, node.hash, .little);
        if (node.is_leaf) {
            try writeBoolFlag(writer, node.segment != null);
            if (node.segment) |seg| {
                try writeSegment(writer, seg);
            }
            var chain_len: usize = 0;
            var chain = node.collision_chain;
            while (chain) |c| {
                chain_len += 1;
                chain = c.next;
            }
            try writer.writeInt(u64, @as(u64, chain_len), .little);
            chain = node.collision_chain;
            while (chain) |c| {
                try writeSegment(writer, c.seg);
                chain = c.next;
            }
            return;
        }
        const children = node.children orelse return error.InvalidNodeState;
        try writer.writeInt(u64, @as(u64, children.len), .little);
        for (children) |maybe_child| {
            try writeBoolFlag(writer, maybe_child != null);
            if (maybe_child) |child| {
                try serializeNode(child, writer);
            }
        }
    }

    fn deserializeNode(allocator: Allocator, reader: anytype) !*BucketNode {
        const is_leaf = try readBoolFlag(reader);
        const height_raw = try reader.readInt(u64, .little);
        if (height_raw > std.math.maxInt(usize)) return error.InvalidData;
        const height: usize = @intCast(height_raw);
        const stored_hash = try reader.readInt(u64, .little);
        const node = try allocator.create(BucketNode);
        var cleanup = true;
        errdefer {
            if (cleanup) {
                recursiveDeinit(node, allocator);
            }
        }
        node.* = try BucketNode.init(allocator, if (is_leaf) 0 else height);
        if (node.is_leaf != is_leaf) {
            return error.InvalidData;
        }
        if (is_leaf) {
            const has_segment = try readBoolFlag(reader);
            if (has_segment) {
                node.segment = try readSegment(allocator, reader);
            }
            const chain_len_raw = try reader.readInt(u64, .little);
            if (chain_len_raw > std.math.maxInt(usize)) return error.InvalidData;
            const chain_len: usize = @intCast(chain_len_raw);
            var head: ?*CollisionNode = null;
            var tail: ?*CollisionNode = null;
            var index: usize = 0;
            while (index < chain_len) : (index += 1) {
                const collision = try allocator.create(CollisionNode);
                collision.* = .{
                    .seg = try readSegment(allocator, reader),
                    .next = null,
                };
                if (head == null) {
                    head = collision;
                    tail = collision;
                } else {
                    tail.?.next = collision;
                    tail = collision;
                }
            }
            node.collision_chain = head;
        } else {
            const children_len_raw = try reader.readInt(u64, .little);
            if (children_len_raw > std.math.maxInt(usize)) return error.InvalidData;
            const children_len: usize = @intCast(children_len_raw);
            if (children_len != bucket_count) {
                return error.InvalidData;
            }
            for (0..children_len) |i| {
                const has_child = try readBoolFlag(reader);
                if (has_child) {
                    node.children.?[i] = try deserializeNode(allocator, reader);
                }
            }
        }
        refreshHash(node);
        if (node.hash != stored_hash) {
            return error.InvalidData;
        }
        cleanup = false;
        return node;
    }

    pub fn serialize(self: *SSI, writer: anytype) !void {
        try writer.writeInt(u64, @as(u64, self.max_height), .little);
        try writer.writeInt(u64, @as(u64, self.height), .little);
        try writer.writeInt(u64, @as(u64, self.size), .little);
        try writeBoolFlag(writer, self.root != null);
        if (self.root) |root| {
            try serializeNode(root, writer);
        }
    }

    pub fn deserialize(allocator: Allocator, reader: anytype) !SSI {
        var ssi = SSI.init(allocator);
        const max_height_raw = try reader.readInt(u64, .little);
        const height_raw = try reader.readInt(u64, .little);
        const size_raw = try reader.readInt(u64, .little);
        if (max_height_raw > std.math.maxInt(usize) or height_raw > std.math.maxInt(usize) or size_raw > std.math.maxInt(usize)) {
            return error.InvalidData;
        }
        ssi.max_height = @intCast(max_height_raw);
        ssi.height = @intCast(height_raw);
        ssi.size = @intCast(size_raw);
        const has_root = try readBoolFlag(reader);
        if (has_root) {
            ssi.root = try deserializeNode(allocator, reader);
        }
        if (ssi.countSegments() != ssi.size) {
            ssi.deinit();
            return error.InvalidData;
        }
        return ssi;
    }

    pub fn exportToTensor(self: *SSI, allocator: Allocator) !Tensor {
        const segment_count = self.countSegments();
        const rows = if (segment_count == 0) 1 else segment_count;
        var tensor = try Tensor.init(allocator, &.{ rows, tensor_width });
        @memset(tensor.data, 0);
        const root = self.root orelse return tensor;
        var row: usize = 0;
        if (root.children) |children| {
            for (children) |maybe_child| {
                if (maybe_child) |leaf| {
                    if (leaf.segment) |seg| {
                        encodeSegmentRow(&tensor, row, seg);
                        row += 1;
                    }
                    var chain = leaf.collision_chain;
                    while (chain) |c| {
                        encodeSegmentRow(&tensor, row, c.seg);
                        row += 1;
                        chain = c.next;
                    }
                }
            }
        }
        return tensor;
    }

    fn encodeSegmentRow(tensor: *Tensor, row: usize, seg: Segment) void {
        const offset = row * tensor_width;
        tensor.data[offset + 0] = safeFloat(@as(f32, @floatFromInt(seg.tokens.len)));
        tensor.data[offset + 1] = safeFloat(bitsToFloat(low32(seg.position)));
        tensor.data[offset + 2] = safeFloat(bitsToFloat(high32(seg.position)));
        tensor.data[offset + 3] = safeFloat(seg.score);
        tensor.data[offset + 4] = safeFloat(bitsToFloat(low32(seg.anchor_hash)));
        tensor.data[offset + 5] = safeFloat(bitsToFloat(high32(seg.anchor_hash)));
        var i: usize = 0;
        while (i < seg.tokens.len and i < 128) : (i += 1) {
            tensor.data[offset + 6 + i] = safeFloat(std.math.clamp(bitsToFloat(seg.tokens[i]), -3.4e38, 3.4e38));
        }
    }

    pub fn importFromTensor(self: *SSI, tensor: *const Tensor) !void {
        self.deinit();
        if (tensor.shape.dims.len < 2) {
            return;
        }
        if (tensor.shape.dims[1] < tensor_width) {
            return error.InvalidData;
        }
        const rows = tensor.shape.dims[0];
        var tokens_buffer: [128]u32 = undefined;
        var row: usize = 0;
        while (row < rows) : (row += 1) {
            const offset = row * tensor_width;
            if (offset + tensor_width > tensor.data.len) {
                break;
            }
            const token_len_float = tensor.data[offset + 0];
            if (!(token_len_float >= 0) or std.math.isInf(token_len_float)) {
                continue;
            }
            const token_len_raw: usize = @intFromFloat(token_len_float);
            const token_len = @min(token_len_raw, 128);
            const position = joinU64(floatToBits(tensor.data[offset + 1]), floatToBits(tensor.data[offset + 2]));
            const score = tensor.data[offset + 3];
            const anchor_hash = joinU64(floatToBits(tensor.data[offset + 4]), floatToBits(tensor.data[offset + 5]));
            var i: usize = 0;
            while (i < token_len) : (i += 1) {
                tokens_buffer[i] = floatToBits(tensor.data[offset + 6 + i]);
            }
            try self.addSequenceWithMetadata(tokens_buffer[0..token_len], position, score, anchor_hash);
        }
    }

    pub fn merge(self: *SSI, other: *const SSI) !void {
        try other.copyInto(self);
    }

    pub fn split(self: *SSI, threshold: f32) !SSI {
        var result = SSI.init(self.allocator);
        result.max_height = self.max_height;
        if (self.root == null) {
            return result;
        }
        const root = self.root.?;
        if (root.children) |children| {
            for (children) |maybe_child| {
                if (maybe_child) |leaf| {
                    if (leaf.segment) |seg| {
                        if (seg.score > threshold) {
                            try result.addSequenceWithMetadata(seg.tokens, seg.position, seg.score, seg.anchor_hash);
                        }
                    }
                    var chain = leaf.collision_chain;
                    while (chain) |c| {
                        if (c.seg.score > threshold) {
                            try result.addSequenceWithMetadata(c.seg.tokens, c.seg.position, c.seg.score, c.seg.anchor_hash);
                        }
                        chain = c.next;
                    }
                }
            }
        }
        return result;
    }

    pub fn balance(self: *SSI) void {
        if (self.root == null) {
            return;
        }
        var rebuilt = SSI.init(self.allocator);
        rebuilt.max_height = self.max_height;
        self.copyInto(&rebuilt) catch {
            rebuilt.deinit();
            return;
        };
        if (rebuilt.countSegments() != self.size) {
            rebuilt.deinit();
            return;
        }
        const old_root = self.root;
        const old_height = self.height;
        const old_size = self.size;
        self.root = rebuilt.root;
        self.height = rebuilt.height;
        self.size = rebuilt.size;
        rebuilt.root = old_root;
        rebuilt.height = old_height;
        rebuilt.size = old_size;
        rebuilt.deinit();
    }

    pub fn stats(self: *const SSI) struct { nodes: usize, leaves: usize, depth: usize } {
        var nodes: usize = 0;
        var leaves: usize = 0;
        var depth: usize = 0;
        const root = self.root orelse return .{ .nodes = 0, .leaves = 0, .depth = 0 };
        var stack = std.ArrayList(struct { node: *const BucketNode, d: usize }).init(self.allocator);
        defer stack.deinit();
        stack.append(.{ .node = root, .d = 0 }) catch return .{ .nodes = nodes, .leaves = leaves, .depth = depth };
        while (stack.pop()) |entry| {
            nodes += 1;
            if (entry.node.is_leaf) {
                leaves += 1;
            }
            if (entry.d > depth) {
                depth = entry.d;
            }
            if (entry.node.children) |children| {
                for (children) |maybe_child| {
                    if (maybe_child) |child| {
                        stack.append(.{ .node = child, .d = entry.d + 1 }) catch {};
                    }
                }
            }
        }
        return .{ .nodes = nodes, .leaves = leaves, .depth = depth };
    }

    fn validateLeaf(node: *const BucketNode) bool {
        if (!node.is_leaf) {
            return false;
        }
        if (node.height != 0) {
            return false;
        }
        if (node.children != null) {
            return false;
        }
        if (node.segment == null and node.collision_chain == null) {
            return true;
        }
        return computeLeafHash(node) == node.hash;
    }

    fn validateNode(node: *const BucketNode, position_set: anytype) !bool {
        if (node.is_leaf) {
            if (!validateLeaf(node)) {
                return false;
            }
            if (node.segment) |seg| {
                if (position_set.contains(seg.position)) return false;
                try position_set.put(seg.position, {});
            }
            var chain = node.collision_chain;
            while (chain) |c| {
                if (position_set.contains(c.seg.position)) return false;
                try position_set.put(c.seg.position, {});
                chain = c.next;
            }
            return true;
        }
        if (node.height != bucket_width) {
            return false;
        }
        const children = node.children orelse return false;
        if (children.len != bucket_count) {
            return false;
        }
        var acc: u64 = 0;
        for (children) |maybe_child| {
            if (maybe_child) |child| {
                if (!(try validateNode(child, position_set))) {
                    return false;
                }
                acc +%= child.hash;
            }
        }
        return acc == node.hash;
    }

    pub fn validate(self: *SSI) bool {
        const root = self.root orelse return self.size == 0;
        if (self.height != bucket_width) {
            return false;
        }
        var position_set = std.AutoHashMap(u64, void).init(self.allocator);
        defer position_set.deinit();
        const valid = validateNode(root, &position_set) catch return false;
        if (!valid) return false;
        const counted = position_set.count();
        if (counted != self.size) {
            return false;
        }
        return true;
    }
};



pub const RankerConfig = struct {
    pub const STREAMING_BUFFER_SIZE: usize = 1024;
    pub const STREAMING_WINDOW_SIZE: usize = 512;
    pub const DEFAULT_TOP_N_RETRIEVAL: usize = 1000;
    pub const HASH_SEED_MULTIPLIER_A: u64 = 0x9e3779b97f4a7c15;
    pub const HASH_SEED_MULTIPLIER_B: u64 = 0x517cc1b727220a95;
    pub const LEARNING_RATE: f32 = 0.01;
    pub const DIVERSITY_WEIGHT: f32 = 0.3;
    pub const PROXIMITY_WEIGHT: f32 = 0.3;
    pub const MAX_RAW_SCORE: f32 = 100.0;
    pub const BASE_SCORE_WEIGHT: f32 = 0.4;
    pub const OVERLAP_WEIGHT: f32 = 0.3;
    pub const JACCARD_WEIGHT: f32 = 0.3;
    pub const SCORE_RETRIEVAL_LIMIT: usize = 32;
    pub const MAX_NGRAM_ORDER: usize = 64;
    pub const PROXIMITY_SCALE: u64 = 1024;
    pub const SCORE_SIGMOID_CENTER: f32 = 0.8;
    pub const SCORE_SIGMOID_WIDTH: f32 = 0.4;
};

fn tokenToLEBytes(token: u32) [4]u8 {
    return mem.toBytes(mem.nativeToLittle(u32, token));
}

fn encodeNgramLE(ngram: []const u32, buf: []u8) void {
    for (ngram, 0..) |token, i| {
        const le = tokenToLEBytes(token);
        buf[i * 4 + 0] = le[0];
        buf[i * 4 + 1] = le[1];
        buf[i * 4 + 2] = le[2];
        buf[i * 4 + 3] = le[3];
    }
}

fn sigmoidScale(raw: f32) f32 {
    const z = (raw - RankerConfig.SCORE_SIGMOID_CENTER) / RankerConfig.SCORE_SIGMOID_WIDTH;
    if (z >= 0.0) {
        return 1.0 / (1.0 + @exp(-z));
    } else {
        const e = @exp(z);
        return e / (1.0 + e);
    }
}

fn sigmoidDerivative(out: f32) f32 {
    return out * (1.0 - out) / RankerConfig.SCORE_SIGMOID_WIDTH;
}

fn freeRankedSegments(segments: []RankedSegment, allocator: Allocator) void {
    for (segments) |*seg| seg.deinit(allocator);
    allocator.free(segments);
}

const RankedSegmentComparator = struct {
    pub fn lessThan(_: void, a: RankedSegment, b: RankedSegment) std.math.Order {
        if (math.isNan(a.score) and math.isNan(b.score)) return .eq;
        if (math.isNan(a.score)) return .lt;
        if (math.isNan(b.score)) return .gt;
        return std.math.order(a.score, b.score);
    }
};

pub const Ranker = struct {
    ngram_weights: []f32,
    lsh_hash_params: []u64,
    num_hash_functions: usize,
    num_ngrams: usize,
    seed: u64,
    allocator: Allocator,

    pub fn init(allocator: Allocator, num_ngrams: usize, num_hash_funcs: usize, seed: u64) !Ranker {
        if (num_ngrams == 0) return error.InvalidParameter;
        if (num_ngrams > RankerConfig.MAX_NGRAM_ORDER) return error.InvalidParameter;
        if (num_hash_funcs == 0) return error.InvalidParameter;

        const weights = try allocator.alloc(f32, num_ngrams);
        errdefer allocator.free(weights);

        var i: usize = 0;
        while (i < weights.len) : (i += 1) {
            const decay = 1.0 / @as(f32, @floatFromInt(i + 1));
            weights[i] = decay;
        }

        const hash_params = try allocator.alloc(u64, num_hash_funcs * 2);
        errdefer allocator.free(hash_params);

        i = 0;
        while (i < num_hash_funcs) : (i += 1) {
            const i_u64: u64 = @intCast(i);
            const i_plus_one: u64 = @intCast(i + 1);
            hash_params[i * 2] = seed +% (i_u64 *% RankerConfig.HASH_SEED_MULTIPLIER_A);
            hash_params[i * 2 + 1] = seed +% (i_plus_one *% RankerConfig.HASH_SEED_MULTIPLIER_B);
        }

        return .{
            .ngram_weights = weights,
            .lsh_hash_params = hash_params,
            .num_hash_functions = num_hash_funcs,
            .num_ngrams = num_ngrams,
            .seed = seed,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Ranker) void {
        self.allocator.free(self.ngram_weights);
        self.allocator.free(self.lsh_hash_params);
    }

    fn windowHash(self: *const Ranker, window: []const u32) u64 {
        const cap_tokens = @min(window.len, RankerConfig.MAX_NGRAM_ORDER);
        var buf: [RankerConfig.MAX_NGRAM_ORDER * 4]u8 = undefined;
        encodeNgramLE(window[0..cap_tokens], buf[0 .. cap_tokens * 4]);
        return stableHash(buf[0 .. cap_tokens * 4], self.seed);
    }

    fn baseRaw(self: *const Ranker, tokens: []const u32, candidate_pos: u64, exclude_pos: ?u64, retrieved: []const RankedSegment, grad_buf: ?[]f32, allocator: Allocator) !f32 {
        if (tokens.len == 0) return 0.0;

        const gram_limit = @min(self.num_ngrams, tokens.len);
        const rates = try allocator.alloc(f32, gram_limit);
        defer allocator.free(rates);

        var ngram_score: f32 = 0.0;
        var weight_sum: f32 = 0.0;
        var gram: usize = 1;
        while (gram <= gram_limit) : (gram += 1) {
            const weight_idx = @min(gram - 1, self.ngram_weights.len - 1);
            const w = self.ngram_weights[weight_idx];
            const cand_windows: usize = tokens.len - gram + 1;
            var ref_count: usize = retrieved.len;
            if (exclude_pos) |ep| {
                ref_count = 0;
                for (retrieved) |seg| {
                    if (seg.position == ep) continue;
                    ref_count += 1;
                }
            }
            const total_compare: usize = cand_windows * @max(ref_count, 1);

            var token_windows = std.AutoHashMap(u64, void).init(allocator);
            defer token_windows.deinit();

            var ti: usize = 0;
            while (ti + gram <= tokens.len) : (ti += 1) {
                try token_windows.put(self.windowHash(tokens[ti .. ti + gram]), {});
            }

            var matches_total: usize = 0;
            for (retrieved) |seg| {
                if (exclude_pos) |ep| {
                    if (seg.position == ep) continue;
                }
                const sim = seg.score;
                if (math.isNan(sim) or math.isInf(sim)) continue;
                var matches: usize = 0;
                var si: usize = 0;
                while (si + gram <= seg.tokens.len) : (si += 1) {
                    if (token_windows.contains(self.windowHash(seg.tokens[si .. si + gram]))) {
                        matches += 1;
                    }
                }
                matches_total += matches;
            }

            var rate: f32 = 0.0;
            if (total_compare > 0) {
                rate = @as(f32, @floatFromInt(matches_total)) / @as(f32, @floatFromInt(total_compare));
            }
            rates[gram - 1] = rate;
            weight_sum += w;
            ngram_score += w * rate;
        }

        const ngram_norm = if (weight_sum > 0.0) ngram_score / weight_sum else 0.0;
        const diversity_score = try self.computeTokenDiversity(tokens, allocator);
        const proximity = self.anchorProximity(candidate_pos, exclude_pos, retrieved);
        const raw = ngram_norm + RankerConfig.DIVERSITY_WEIGHT * diversity_score + RankerConfig.PROXIMITY_WEIGHT * proximity;

        if (grad_buf) |gb| {
            if (weight_sum > 0.0) {
                var gi: usize = 0;
                while (gi < gram_limit) : (gi += 1) {
                    gb[gi] = (rates[gi] - ngram_norm) / weight_sum;
                }
            }
            if (gram_limit < gb.len) {
                @memset(gb[gram_limit..], 0.0);
            }
        }

        return raw;
    }

    fn scoreSequenceRaw(self: *const Ranker, tokens: []const u32, ssi: *const SSI, grad_buf: ?[]f32, allocator: Allocator) !f32 {
        const retrieved = try ssi.retrieveTopK(tokens, RankerConfig.SCORE_RETRIEVAL_LIMIT, allocator);
        defer freeRankedSegments(retrieved, allocator);
        return self.baseRaw(tokens, 0, null, retrieved, grad_buf, allocator);
    }

    fn scoreSequenceAlloc(self: *const Ranker, tokens: []const u32, ssi: *const SSI, allocator: Allocator) !f32 {
        const retrieved = try ssi.retrieveTopK(tokens, RankerConfig.SCORE_RETRIEVAL_LIMIT, allocator);
        defer freeRankedSegments(retrieved, allocator);
        const raw = try self.baseRaw(tokens, 0, null, retrieved, null, allocator);
        return sigmoidScale(raw);
    }

    pub fn scoreSequence(self: *const Ranker, tokens: []const u32, ssi: *const SSI) !f32 {
        return self.scoreSequenceAlloc(tokens, ssi, self.allocator);
    }

    fn combinedScore(self: *const Ranker, tokens: []const u32, query: []const u32, candidate_pos: u64, exclude_pos: ?u64, retrieved: []const RankedSegment, grad_buf: ?[]f32, allocator: Allocator) !f32 {
        const raw = try self.baseRaw(tokens, candidate_pos, exclude_pos, retrieved, grad_buf, allocator);
        const base_scaled = sigmoidScale(raw);
        if (query.len == 0) return base_scaled;

        var overlap: f32 = 0.0;
        var jaccard: f32 = 0.0;
        if (tokens.len > 0 and query.len > 0) {
            const m = try self.setMetrics(tokens, query, allocator);
            overlap = @as(f32, @floatFromInt(m.intersection)) / @as(f32, @floatFromInt(@max(@min(m.card_a, m.card_b), 1)));
            const denom = m.card_a + m.card_b - m.intersection;
            jaccard = @as(f32, @floatFromInt(m.intersection)) / @as(f32, @floatFromInt(@max(denom, 1)));
        } else if (tokens.len == 0 and query.len == 0) {
            jaccard = 1.0;
        }

        return math.clamp(base_scaled * RankerConfig.BASE_SCORE_WEIGHT + overlap * RankerConfig.OVERLAP_WEIGHT + jaccard * RankerConfig.JACCARD_WEIGHT, 0.0, 1.0);
    }

    fn scoreSequenceWithQueryAlloc(self: *const Ranker, tokens: []const u32, query: []const u32, ssi: *const SSI, allocator: Allocator) !f32 {
        const retrieved = try ssi.retrieveTopK(query, RankerConfig.SCORE_RETRIEVAL_LIMIT, allocator);
        defer freeRankedSegments(retrieved, allocator);
        return self.combinedScore(tokens, query, 0, null, retrieved, null, allocator);
    }

    pub fn scoreSequenceWithQuery(self: *const Ranker, tokens: []const u32, query: []const u32, ssi: *const SSI) !f32 {
        return self.scoreSequenceWithQueryAlloc(tokens, query, ssi, self.allocator);
    }

    fn computeTokenDiversity(self: *const Ranker, tokens: []const u32, allocator: Allocator) !f32 {
        _ = self;
        if (tokens.len == 0) return 0.0;

        var unique_tokens = std.AutoHashMap(u32, void).init(allocator);
        defer unique_tokens.deinit();

        for (tokens) |token| {
            try unique_tokens.put(token, {});
        }

        const unique_count = unique_tokens.count();
        const diversity = @as(f32, @floatFromInt(unique_count)) / @as(f32, @floatFromInt(tokens.len));

        return diversity;
    }

    fn setMetrics(self: *const Ranker, tokens: []const u32, query: []const u32, allocator: Allocator) !struct { intersection: usize, card_a: usize, card_b: usize } {
        _ = self;
        var set_a = std.AutoHashMap(u32, void).init(allocator);
        defer set_a.deinit();

        for (tokens) |token| {
            try set_a.put(token, {});
        }

        var set_b = std.AutoHashMap(u32, void).init(allocator);
        defer set_b.deinit();

        for (query) |qtoken| {
            try set_b.put(qtoken, {});
        }

        var intersection: usize = 0;
        var it = set_a.keyIterator();
        while (it.next()) |key| {
            if (set_b.contains(key.*)) {
                intersection += 1;
            }
        }

        return .{
            .intersection = intersection,
            .card_a = set_a.count(),
            .card_b = set_b.count(),
        };
    }

    fn computeTokenOverlap(self: *const Ranker, tokens: []const u32, query: []const u32, allocator: Allocator) !f32 {
        if (tokens.len == 0 or query.len == 0) return 0.0;

        const m = try self.setMetrics(tokens, query, allocator);
        const denom = @min(m.card_a, m.card_b);
        if (denom == 0) return 0.0;
        return @as(f32, @floatFromInt(m.intersection)) / @as(f32, @floatFromInt(denom));
    }

    fn computeJaccardSimilarity(self: *const Ranker, tokens: []const u32, query: []const u32, allocator: Allocator) !f32 {
        if (tokens.len == 0 and query.len == 0) return 1.0;
        if (tokens.len == 0 or query.len == 0) return 0.0;

        const m = try self.setMetrics(tokens, query, allocator);
        const denom = m.card_a + m.card_b - m.intersection;
        if (denom == 0) return 0.0;
        return @as(f32, @floatFromInt(m.intersection)) / @as(f32, @floatFromInt(denom));
    }

    fn anchorProximity(self: *const Ranker, candidate_pos: u64, exclude_pos: ?u64, retrieved: []const RankedSegment) f32 {
        _ = self;
        if (retrieved.len == 0) return 0.0;

        var anchors: usize = 0;
        var total_dist: u64 = 0;
        for (retrieved) |seg| {
            if (exclude_pos) |ep| {
                if (seg.position == ep) continue;
            }
            if (!seg.anchor) continue;
            anchors += 1;
            const d = if (seg.position >= candidate_pos) seg.position - candidate_pos else candidate_pos - seg.position;
            total_dist += @min(d, RankerConfig.PROXIMITY_SCALE);
        }
        if (anchors == 0) return 0.0;
        const mean = total_dist / @as(u64, @intCast(anchors));
        return 1.0 - math.clamp(@as(f32, @floatFromInt(mean)) / @as(f32, @floatFromInt(RankerConfig.PROXIMITY_SCALE)), 0.0, 1.0);
    }

    pub fn rankCandidates(self: *const Ranker, candidates: []RankedSegment, ssi: *const SSI, allocator: Allocator) !void {
        return self.rankCandidatesWithQuery(candidates, &[_]u32{}, ssi, allocator);
    }

    fn rearrangeCandidatesByIndices(candidates: []RankedSegment, indices: []const usize, scores: []const f32, allocator: Allocator) !void {
        if (candidates.len == 0) return;
        if (indices.len != candidates.len) return error.LengthMismatch;
        if (scores.len != candidates.len) return error.LengthMismatch;

        const copy = try allocator.alloc(RankedSegment, candidates.len);
        defer allocator.free(copy);

        for (candidates, 0..) |c, idx| {
            copy[idx] = c;
        }

        var i: usize = 0;
        while (i < candidates.len) : (i += 1) {
            const src = indices[i];
            candidates[i] = copy[src];
            candidates[i].score = scores[src];
        }
    }

    fn sortCandidatesByScore(candidates: []RankedSegment, scores: []const f32, allocator: Allocator) !void {
        if (candidates.len == 0) return;

        const indices = try allocator.alloc(usize, candidates.len);
        defer allocator.free(indices);

        var i: usize = 0;
        while (i < candidates.len) : (i += 1) {
            indices[i] = i;
        }

        const SortContext = struct {
            scores: []const f32,
            pub fn lessThan(ctx: @This(), a: usize, b: usize) bool {
                const score_a = ctx.scores[a];
                const score_b = ctx.scores[b];
                if (math.isNan(score_a)) return false;
                if (math.isNan(score_b)) return true;
                return score_a > score_b;
            }
        };
        std.mem.sort(usize, indices, SortContext{ .scores = scores }, SortContext.lessThan);

        try rearrangeCandidatesByIndices(candidates, indices, scores, allocator);
    }

    pub fn rankCandidatesWithQuery(self: *const Ranker, candidates: []RankedSegment, query: []const u32, ssi: *const SSI, allocator: Allocator) !void {
        if (candidates.len == 0) return;

        const scores = try allocator.alloc(f32, candidates.len);
        defer allocator.free(scores);

        if (query.len > 0) {
            const retrieved = try ssi.retrieveTopK(query, RankerConfig.SCORE_RETRIEVAL_LIMIT, allocator);
            defer freeRankedSegments(retrieved, allocator);
            var i: usize = 0;
            while (i < candidates.len) : (i += 1) {
                scores[i] = try self.combinedScore(candidates[i].tokens, query, candidates[i].position, candidates[i].position, retrieved, null, allocator);
            }
        } else {
            var i: usize = 0;
            while (i < candidates.len) : (i += 1) {
                const retrieved = try ssi.retrieveTopK(candidates[i].tokens, RankerConfig.SCORE_RETRIEVAL_LIMIT, allocator);
                scores[i] = self.combinedScore(candidates[i].tokens, &.{}, candidates[i].position, candidates[i].position, retrieved, null, allocator) catch |err| {
                    freeRankedSegments(retrieved, allocator);
                    return err;
                };
                freeRankedSegments(retrieved, allocator);
            }
        }

        try sortCandidatesByScore(candidates, scores, allocator);
    }

    pub fn batchScore(self: *const Ranker, sequences: []const []const u32, ssi: *const SSI, allocator: Allocator) ![]f32 {
        if (sequences.len == 0) return allocator.alloc(f32, 0);

        const batch_size = sequences.len;
        const scores = try allocator.alloc(f32, batch_size);
        errdefer allocator.free(scores);

        var b: usize = 0;
        while (b < batch_size) : (b += 1) {
            scores[b] = try self.scoreSequenceAlloc(sequences[b], ssi, allocator);
        }
        return scores;
    }

    pub fn topKHeap(self: *const Ranker, ssi: *const SSI, query: []const u32, k: usize, allocator: Allocator) ![]RankedSegment {
        if (k == 0) return allocator.alloc(RankedSegment, 0);

        const retrieval_count = @max(k, RankerConfig.DEFAULT_TOP_N_RETRIEVAL);

        var heap = std.PriorityQueue(RankedSegment, void, RankedSegmentComparator.lessThan).init(allocator, {});
        defer {
            while (heap.removeOrNull()) |item| {
                var m = item;
                m.deinit(allocator);
            }
            heap.deinit();
        }

        const candidates = try ssi.retrieveTopK(query, retrieval_count, allocator);
        defer freeRankedSegments(candidates, allocator);

        const reference_len = @min(candidates.len, RankerConfig.SCORE_RETRIEVAL_LIMIT);
        const reference = candidates[0..reference_len];

        var i: usize = 0;
        while (i < candidates.len) : (i += 1) {
            const cand = candidates[i];
            const score = try self.combinedScore(cand.tokens, query, cand.position, cand.position, reference, null, allocator);

            if (math.isNan(score) or math.isInf(score)) continue;

            if (heap.count() < k) {
                var ranked = try RankedSegment.init(allocator, cand.tokens, score, cand.position, cand.anchor);
                heap.add(ranked) catch |err| {
                    ranked.deinit(allocator);
                    return err;
                };
            } else if (heap.peek()) |top| {
                if (math.isNan(top.score) or score > top.score) {
                    var removed = heap.remove();
                    removed.deinit(allocator);
                    var ranked = try RankedSegment.init(allocator, cand.tokens, score, cand.position, cand.anchor);
                    heap.add(ranked) catch |err| {
                        ranked.deinit(allocator);
                        return err;
                    };
                }
            }
        }

        const result_count = heap.count();
        const top_n = try allocator.alloc(RankedSegment, result_count);
        var n_placed: usize = 0;
        errdefer {
            var j: usize = 0;
            while (j < n_placed) : (j += 1) {
                top_n[result_count - 1 - j].deinit(allocator);
            }
            allocator.free(top_n);
        }

        var idx: usize = result_count;
        while (heap.removeOrNull()) |item| {
            if (idx > 0) {
                idx -= 1;
                top_n[idx] = item;
                n_placed += 1;
            } else {
                var mutable_item = item;
                mutable_item.deinit(allocator);
            }
        }

        return top_n;
    }

    pub fn updateWeights(self: *Ranker, gradients: []const f32) void {
        var i: usize = 0;
        while (i < @min(self.ngram_weights.len, gradients.len)) : (i += 1) {
            const grad = gradients[i];
            if (math.isNan(grad) or math.isInf(grad)) continue;
            self.ngram_weights[i] -= RankerConfig.LEARNING_RATE * grad;
            self.ngram_weights[i] = math.clamp(self.ngram_weights[i], 0.0, 1.0);
        }
    }

    pub fn minHashSignature(self: *const Ranker, tokens: []const u32) ![]u64 {
        if (tokens.len == 0) {
            const sig = try self.allocator.alloc(u64, self.num_hash_functions);
            @memset(sig, std.math.maxInt(u64));
            return sig;
        }

        const sig = try self.allocator.alloc(u64, self.num_hash_functions);
        errdefer self.allocator.free(sig);

        const token_hashes = try self.allocator.alloc(u64, tokens.len);
        defer self.allocator.free(token_hashes);

        var ti: usize = 0;
        while (ti < tokens.len) : (ti += 1) {
            const le_bytes = tokenToLEBytes(tokens[ti]);
            token_hashes[ti] = stableHash(&le_bytes, self.seed);
        }

        var h: usize = 0;
        while (h < self.num_hash_functions) : (h += 1) {
            var min_hash: u64 = std.math.maxInt(u64);
            const seed_a = self.lsh_hash_params[h * 2];
            const seed_b = self.lsh_hash_params[h * 2 + 1];
            for (token_hashes) |th| {
                const hash_val = (th ^ seed_a) *% RankerConfig.HASH_SEED_MULTIPLIER_B +% seed_b;
                if (hash_val < min_hash) {
                    min_hash = hash_val;
                }
            }
            sig[h] = min_hash;
        }
        return sig;
    }

    pub fn jaccardSimilarityFromSignatures(sig1: []const u64, sig2: []const u64) f32 {
        if (sig1.len != sig2.len) return 0.0;
        if (sig1.len == 0) return 0.0;

        var matches: usize = 0;
        var i: usize = 0;
        while (i < sig1.len) : (i += 1) {
            if (sig1[i] == sig2[i]) {
                matches += 1;
            }
        }
        return @as(f32, @floatFromInt(matches)) / @as(f32, @floatFromInt(sig1.len));
    }

    pub fn minHashBitmask(self: *const Ranker, tokens: []const u32) ![]u64 {
        const words = (self.num_hash_functions + 63) / 64;
        const mask = try self.allocator.alloc(u64, words);
        errdefer self.allocator.free(mask);
        @memset(mask, 0);

        const sig = try self.minHashSignature(tokens);
        defer self.allocator.free(sig);

        var h: usize = 0;
        while (h < self.num_hash_functions) : (h += 1) {
            if ((sig[h] & 1) != 0) {
                mask[h / 64] |= @as(u64, 1) << @intCast(h % 64);
            }
        }
        return mask;
    }

    pub fn jaccardFromBitmasks(mask1: []const u64, mask2: []const u64, valid_bits: usize) f32 {
        if (mask1.len != mask2.len) return 0.0;
        if (mask1.len == 0 or valid_bits == 0) return 0.0;

        const vector_len: usize = 4;
        const full_words = valid_bits / 64;
        var matches: usize = 0;
        var w: usize = 0;
        while (w + vector_len <= full_words) : (w += vector_len) {
            const a: @Vector(vector_len, u64) = mask1[w..][0..vector_len].*;
            const b: @Vector(vector_len, u64) = mask2[w..][0..vector_len].*;
            const agree = ~(a ^ b);
            var v: usize = 0;
            while (v < vector_len) : (v += 1) {
                matches += @popCount(agree[v]);
            }
        }
        while (w < full_words) : (w += 1) {
            matches += @popCount(~(mask1[w] ^ mask2[w]));
        }
        if (full_words < mask1.len) {
            const rem = valid_bits % 64;
            const valid_mask: u64 = if (rem == 0) std.math.maxInt(u64) else (@as(u64, 1) << @intCast(rem)) - 1;
            matches += @popCount((~(mask1[full_words] ^ mask2[full_words])) & valid_mask);
        }
        const denom = valid_bits;
        if (denom == 0) return 0.0;
        const ratio = @as(f32, @floatFromInt(matches)) / @as(f32, @floatFromInt(denom));
        const estimate = 2.0 * ratio - 1.0;
        return math.clamp(estimate, 0.0, 1.0);
    }

    pub fn jaccardSignatureBitmask(self: *const Ranker, tokens1: []const u32, tokens2: []const u32) !f32 {
        const mask1 = try self.minHashBitmask(tokens1);
        defer self.allocator.free(mask1);
        const mask2 = try self.minHashBitmask(tokens2);
        defer self.allocator.free(mask2);
        return jaccardFromBitmasks(mask1, mask2, self.num_hash_functions);
    }

    pub fn estimateJaccard(set1: BitSet, set2: BitSet) f32 {
        const len1 = set1.bits.len;
        const len2 = set2.bits.len;
        const max_words = @max(len1, len2);

        if (max_words == 0) return 1.0;

        var intersect: usize = 0;
        var union_count: usize = 0;
        var i: usize = 0;
        while (i < max_words) : (i += 1) {
            const w1: u64 = if (i < len1) set1.bits[i] else 0;
            const w2: u64 = if (i < len2) set2.bits[i] else 0;
            intersect += @popCount(w1 & w2);
            union_count += @popCount(w1 | w2);
        }
        return if (union_count == 0) 1.0 else @as(f32, @floatFromInt(intersect)) / @as(f32, @floatFromInt(union_count));
    }

    pub fn vectorScore(embedding: *const Tensor, query_emb: *const Tensor) !f32 {
        if (!mem.eql(usize, embedding.shape.dims, query_emb.shape.dims)) return Error.ShapeMismatch;
        if (embedding.data.len != query_emb.data.len) return Error.ShapeMismatch;
        if (embedding.data.len == 0) return 0.0;

        var dot_prod: f32 = 0.0;
        var norm_emb: f32 = 0.0;
        var norm_query: f32 = 0.0;

        const len = embedding.data.len;
        var i: usize = 0;
        while (i < len) : (i += 1) {
            const e = embedding.data[i];
            const q = query_emb.data[i];

            if (math.isNan(e) or math.isNan(q)) continue;
            if (math.isInf(e) or math.isInf(q)) continue;

            dot_prod += e * q;
            norm_emb += e * e;
            norm_query += q * q;
        }

        if (norm_emb <= 0.0 or norm_query <= 0.0) return 0.0;

        norm_emb = math.sqrt(norm_emb);
        norm_query = math.sqrt(norm_query);

        const result = dot_prod / (norm_emb * norm_query);
        return math.clamp(result, -1.0, 1.0);
    }

    pub fn dotProductScore(embedding: *const Tensor, query_emb: *const Tensor) !f32 {
        if (!mem.eql(usize, embedding.shape.dims, query_emb.shape.dims)) return Error.ShapeMismatch;
        if (embedding.data.len != query_emb.data.len) return Error.ShapeMismatch;
        if (embedding.data.len == 0) return 0.0;

        var dot_prod: f32 = 0.0;
        const len = embedding.data.len;
        var i: usize = 0;
        while (i < len) : (i += 1) {
            const e = embedding.data[i];
            const q = query_emb.data[i];

            if (math.isNan(e) or math.isNan(q)) continue;
            if (math.isInf(e) or math.isInf(q)) continue;

            dot_prod += e * q;
        }
        return dot_prod;
    }

    pub fn weightedAverage(scores: []const f32, weights: []const f32) !f32 {
        if (scores.len != weights.len) return error.LengthMismatch;
        if (scores.len == 0) return 0.0;

        var num: f32 = 0.0;
        var den: f32 = 0.0;
        var i: usize = 0;
        while (i < scores.len) : (i += 1) {
            const s = scores[i];
            const w = weights[i];

            if (math.isNan(s) or math.isNan(w)) continue;
            if (math.isInf(w) or w < 0.0) return error.InvalidParameter;
            if (math.isInf(s)) continue;

            num += s * w;
            den += w;
        }

        if (den == 0.0) return 0.0;
        return num / den;
    }

    pub fn exponentialDecay(scores: []f32, decay_factor: f32) !void {
        if (scores.len == 0) return;
        if (decay_factor <= 0.0 or decay_factor >= 1.0) return error.InvalidParameter;

        var current_decay: f32 = 1.0;
        var i: usize = 0;
        while (i < scores.len) : (i += 1) {
            if (!math.isNan(scores[i]) and !math.isInf(scores[i])) {
                scores[i] *= current_decay;
            }
            current_decay *= decay_factor;
        }
    }

    pub fn normalizeScores(self: *const Ranker, scores: []f32) void {
        _ = self;
        normalizeScoresStatic(scores);
    }

    fn normalizeScoresStatic(scores: []f32) void {
        if (scores.len == 0) return;

        var min_score: f32 = math.inf(f32);
        var max_score: f32 = -math.inf(f32);
        var valid_count: usize = 0;

        var i: usize = 0;
        while (i < scores.len) : (i += 1) {
            const s = scores[i];
            if (math.isNan(s) or math.isInf(s)) continue;
            valid_count += 1;
            if (s < min_score) min_score = s;
            if (s > max_score) max_score = s;
        }

        if (valid_count == 0) {
            i = 0;
            while (i < scores.len) : (i += 1) {
                scores[i] = 0.0;
            }
            return;
        }
        if (max_score == min_score) {
            i = 0;
            while (i < scores.len) : (i += 1) {
                if (!math.isNan(scores[i]) and !math.isInf(scores[i])) {
                    scores[i] = 0.5;
                } else {
                    scores[i] = 0.0;
                }
            }
            return;
        }

        const range = max_score - min_score;
        i = 0;
        while (i < scores.len) : (i += 1) {
            if (math.isNan(scores[i]) or math.isInf(scores[i])) {
                scores[i] = 0.0;
            } else {
                scores[i] = (scores[i] - min_score) / range;
            }
        }
    }

    pub fn rankByMultipleCriteria(self: *const Ranker, candidates: []RankedSegment, criteria: []const []const f32, weights: []const f32, allocator: Allocator) !void {
        _ = self;
        if (candidates.len == 0) return;
        if (criteria.len == 0) return;
        if (weights.len == 0) return;

        const num_cand = candidates.len;
        const num_crit = @min(criteria.len, weights.len);

        var cr: usize = 0;
        while (cr < num_crit) : (cr += 1) {
            if (criteria[cr].len < num_cand) return error.LengthMismatch;
        }

        const combined = try allocator.alloc(f32, num_cand);
        defer allocator.free(combined);

        var c: usize = 0;
        while (c < num_cand) : (c += 1) {
            var crit_score: f32 = 0.0;
            cr = 0;
            while (cr < num_crit) : (cr += 1) {
                const score_val = criteria[cr][c];
                const weight_val = weights[cr];
                if (!math.isNan(score_val) and !math.isNan(weight_val) and !math.isInf(score_val) and !math.isInf(weight_val)) {
                    crit_score += score_val * weight_val;
                }
            }
            combined[c] = crit_score;
        }

        try sortCandidatesByScore(candidates, combined, allocator);
    }

    pub fn streamingRank(self: *const Ranker, reader: anytype, ssi: *const SSI, k: usize, allocator: Allocator) ![]RankedSegment {
        if (k == 0) return allocator.alloc(RankedSegment, 0);

        var rolling_buffer = std.ArrayList(u32).init(allocator);
        defer rolling_buffer.deinit();

        var heap = std.PriorityQueue(RankedSegment, void, RankedSegmentComparator.lessThan).init(allocator, {});
        defer {
            while (heap.removeOrNull()) |item| {
                var m = item;
                m.deinit(allocator);
            }
            heap.deinit();
        }

        var leftover_bytes: [3]u8 = undefined;
        var leftover_len: usize = 0;
        var position: u64 = 0;
        var read_buf: [RankerConfig.STREAMING_BUFFER_SIZE * @sizeOf(u32)]u8 = undefined;

        while (true) {
            const bytes_read = reader.read(&read_buf) catch |err| return err;
            if (bytes_read == 0) break;

            var combined_buf: []u8 = undefined;
            var combined_len: usize = 0;
            var combined_alloc: ?[]u8 = null;
            defer {
                if (combined_alloc) |ca| allocator.free(ca);
            }

            if (leftover_len > 0) {
                combined_len = leftover_len + bytes_read;
                combined_alloc = try allocator.alloc(u8, combined_len);
                combined_buf = combined_alloc.?;
                var ci: usize = 0;
                while (ci < leftover_len) : (ci += 1) {
                    combined_buf[ci] = leftover_bytes[ci];
                }
                var ri: usize = 0;
                while (ri < bytes_read) : (ri += 1) {
                    combined_buf[leftover_len + ri] = read_buf[ri];
                }
                leftover_len = 0;
            } else {
                combined_buf = read_buf[0..bytes_read];
                combined_len = bytes_read;
            }

            const full_tokens = combined_len / @sizeOf(u32);
            const remainder = combined_len % @sizeOf(u32);

            if (remainder > 0) {
                var ri: usize = 0;
                while (ri < remainder) : (ri += 1) {
                    leftover_bytes[ri] = combined_buf[full_tokens * @sizeOf(u32) + ri];
                }
                leftover_len = remainder;
            }

            var ti: usize = 0;
            while (ti < full_tokens) : (ti += 1) {
                const offset = ti * @sizeOf(u32);
                var token_bytes: [4]u8 = undefined;
                token_bytes[0] = combined_buf[offset + 0];
                token_bytes[1] = combined_buf[offset + 1];
                token_bytes[2] = combined_buf[offset + 2];
                token_bytes[3] = combined_buf[offset + 3];
                const token = mem.readInt(u32, &token_bytes, .little);
                try rolling_buffer.append(token);
            }

            while (rolling_buffer.items.len >= RankerConfig.STREAMING_WINDOW_SIZE) {
                const window = rolling_buffer.items[0..RankerConfig.STREAMING_WINDOW_SIZE];
                const score = try self.scoreSequenceAlloc(window, ssi, allocator);

                if (!math.isNan(score) and !math.isInf(score)) {
                    if (heap.count() < k) {
                        var seg = try RankedSegment.init(allocator, window, score, position, false);
                        heap.add(seg) catch |err| {
                            seg.deinit(allocator);
                            return err;
                        };
                    } else if (heap.peek()) |top| {
                        if (math.isNan(top.score) or score > top.score) {
                            var removed = heap.remove();
                            removed.deinit(allocator);
                            var seg = try RankedSegment.init(allocator, window, score, position, false);
                            heap.add(seg) catch |err| {
                                seg.deinit(allocator);
                                return err;
                            };
                        }
                    }
                }

                const shift = @min(rolling_buffer.items.len, RankerConfig.STREAMING_WINDOW_SIZE / 2);
                const remaining = rolling_buffer.items.len - shift;
                if (remaining > 0) {
                    std.mem.copyForwards(u32, rolling_buffer.items[0..remaining], rolling_buffer.items[shift..rolling_buffer.items.len]);
                }
                rolling_buffer.shrinkRetainingCapacity(remaining);
                position += shift;
            }
        }

        if (leftover_len > 0) return error.InvalidData;

        if (rolling_buffer.items.len > 0) {
            const tail = rolling_buffer.items;
            const score = try self.scoreSequenceAlloc(tail, ssi, allocator);
            if (!math.isNan(score) and !math.isInf(score)) {
                if (heap.count() < k) {
                    var seg = try RankedSegment.init(allocator, tail, score, position, false);
                    heap.add(seg) catch |err| {
                        seg.deinit(allocator);
                        return err;
                    };
                } else if (heap.peek()) |top| {
                    if (math.isNan(top.score) or score > top.score) {
                        var removed = heap.remove();
                        removed.deinit(allocator);
                        var seg = try RankedSegment.init(allocator, tail, score, position, false);
                        heap.add(seg) catch |err| {
                            seg.deinit(allocator);
                            return err;
                        };
                    }
                }
            }
        }

        const result_count = heap.count();
        const result = try allocator.alloc(RankedSegment, result_count);
        var n_placed: usize = 0;
        errdefer {
            var ei: usize = 0;
            while (ei < n_placed) : (ei += 1) {
                result[result_count - 1 - ei].deinit(allocator);
            }
            allocator.free(result);
        }

        var idx: usize = result_count;
        while (heap.removeOrNull()) |item| {
            if (idx > 0) {
                idx -= 1;
                result[idx] = item;
                n_placed += 1;
            } else {
                var m = item;
                m.deinit(allocator);
            }
        }

        return result;
    }

    pub fn parallelScore(self: *const Ranker, sequences: []const []const u32, ssi: *const SSI, num_threads: usize) ![]f32 {
        if (sequences.len == 0) return self.allocator.alloc(f32, 0);

        const scores = try self.allocator.alloc(f32, sequences.len);
        errdefer self.allocator.free(scores);

        if (num_threads <= 1 or sequences.len <= 1) {
            var i: usize = 0;
            while (i < sequences.len) : (i += 1) {
                scores[i] = try self.scoreSequence(sequences[i], ssi);
            }
            return scores;
        }

        const cpu_count = std.Thread.getCpuCount() catch @as(usize, 1);
        const effective_threads = @min(@min(num_threads, sequences.len), cpu_count);
        const chunk_size = sequences.len / effective_threads;
        const remainder_count = sequences.len % effective_threads;

        const ThreadContext = struct {
            ranker: *const Ranker,
            seqs: []const []const u32,
            ssi_ptr: *const SSI,
            out: []f32,
            start: usize,
            end: usize,
            err_flag: bool,
        };

        const contexts = try self.allocator.alloc(ThreadContext, effective_threads);
        defer self.allocator.free(contexts);

        const threads = try self.allocator.alloc(std.Thread, effective_threads);
        defer self.allocator.free(threads);

        const thread_spawned = try self.allocator.alloc(bool, effective_threads);
        defer self.allocator.free(thread_spawned);
        @memset(thread_spawned, false);

        var offset: usize = 0;
        var t: usize = 0;
        while (t < effective_threads) : (t += 1) {
            const this_chunk = chunk_size + @as(usize, if (t < remainder_count) 1 else 0);
            contexts[t] = .{
                .ranker = self,
                .seqs = sequences,
                .ssi_ptr = ssi,
                .out = scores,
                .start = offset,
                .end = offset + this_chunk,
                .err_flag = false,
            };
            offset += this_chunk;
        }

        t = 0;
        while (t < effective_threads) : (t += 1) {
            threads[t] = std.Thread.spawn(.{}, struct {
                fn work(ctx: *ThreadContext) void {
                    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
                    defer arena.deinit();
                    var si: usize = ctx.start;
                    while (si < ctx.end) : (si += 1) {
                        _ = arena.reset(.retain_capacity);
                        ctx.out[si] = ctx.ranker.scoreSequenceAlloc(ctx.seqs[si], ctx.ssi_ptr, arena.allocator()) catch {
                            ctx.err_flag = true;
                            return;
                        };
                    }
                }
            }.work, .{&contexts[t]}) catch {
                var si: usize = contexts[t].start;
                while (si < contexts[t].end) : (si += 1) {
                    scores[si] = self.scoreSequenceAlloc(sequences[si], ssi, self.allocator) catch {
                        contexts[t].err_flag = true;
                        break;
                    };
                }
                thread_spawned[t] = false;
                continue;
            };
            thread_spawned[t] = true;
        }

        t = 0;
        while (t < effective_threads) : (t += 1) {
            if (thread_spawned[t]) {
                threads[t].join();
            }
        }

        var had_error = false;
        t = 0;
        while (t < effective_threads) : (t += 1) {
            if (contexts[t].err_flag) had_error = true;
        }

        if (had_error) {
            return error.ScoringFailed;
        }

        return scores;
    }

    pub fn calibrateWeights(self: *Ranker, training_data: []const []const u32, labels: []const f32, ssi: *const SSI, epochs: usize) !void {
        if (training_data.len == 0 or labels.len == 0) return error.InvalidParameter;
        if (training_data.len != labels.len) return error.LengthMismatch;

        const gradients = try self.allocator.alloc(f32, self.ngram_weights.len);
        defer self.allocator.free(gradients);

        const sample_grad = try self.allocator.alloc(f32, self.ngram_weights.len);
        defer self.allocator.free(sample_grad);

        var epoch: usize = 0;
        while (epoch < epochs) : (epoch += 1) {
            @memset(gradients, 0.0);

            var i: usize = 0;
            while (i < training_data.len) : (i += 1) {
                const sample = training_data[i];
                const label = labels[i];

                @memset(sample_grad, 0.0);
                const raw = try self.scoreSequenceRaw(sample, ssi, sample_grad, self.allocator);
                const pred = sigmoidScale(raw);

                if (math.isNan(pred) or math.isNan(label)) continue;
                if (math.isInf(pred) or math.isInf(label)) continue;

                const err_val = pred - label;
                const d_out = sigmoidDerivative(pred);
                const scale = err_val * d_out;

                var g: usize = 0;
                while (g < gradients.len) : (g += 1) {
                    gradients[g] += sample_grad[g] * scale;
                }
            }

            const n_samples: f32 = @floatFromInt(training_data.len);
            var g: usize = 0;
            while (g < gradients.len) : (g += 1) {
                gradients[g] = gradients[g] / n_samples;
            }

            self.updateWeights(gradients);
        }
    }

    pub fn exportModel(self: *const Ranker, path: []const u8) !void {
        const file = try createFilePath(path, .{});
        defer file.close();
        const writer = file.writer();
        try writer.writeInt(u8, 2, .little);
        try writer.writeInt(u64, @intCast(self.ngram_weights.len), .little);
        try writer.writeInt(u64, @intCast(self.num_ngrams), .little);
        var i: usize = 0;
        while (i < self.ngram_weights.len) : (i += 1) {
            const bits: u32 = @bitCast(self.ngram_weights[i]);
            try writer.writeInt(u32, bits, .little);
        }
        try writer.writeInt(u64, @intCast(self.num_hash_functions), .little);
        i = 0;
        while (i < self.lsh_hash_params.len) : (i += 1) {
            try writer.writeInt(u64, self.lsh_hash_params[i], .little);
        }
        try writer.writeInt(u64, self.seed, .little);
    }

    pub fn importModel(self: *Ranker, path: []const u8) !void {
        const file = try openFilePath(path, .{});
        defer file.close();
        const reader = file.reader();
        const version = try reader.readInt(u8, .little);
        if (version != 2) return error.InvalidVersion;

        const num_w = try reader.readInt(u64, .little);
        const num_ng = try reader.readInt(u64, .little);

        if (num_w == 0 or num_ng == 0) return error.InvalidParameter;
        if (num_w != num_ng) return error.InvalidParameter;
        if (num_w > std.math.maxInt(usize)) return error.InvalidParameter;
        if (num_w > RankerConfig.MAX_NGRAM_ORDER) return error.InvalidParameter;

        const num_w_usize: usize = @intCast(num_w);
        const num_ng_usize: usize = @intCast(num_ng);

        const new_weights = try self.allocator.alloc(f32, num_w_usize);
        errdefer self.allocator.free(new_weights);

        var i: usize = 0;
        while (i < num_w_usize) : (i += 1) {
            const bits = try reader.readInt(u32, .little);
            var weight: f32 = @bitCast(bits);
            if (math.isNan(weight) or math.isInf(weight)) weight = 0.0;
            new_weights[i] = weight;
        }

        const num_h = try reader.readInt(u64, .little);
        if (num_h == 0) return error.InvalidParameter;
        if (num_h > std.math.maxInt(usize) / 2) return error.InvalidParameter;

        const num_h_usize: usize = @intCast(num_h);

        const new_params = try self.allocator.alloc(u64, num_h_usize * 2);
        errdefer self.allocator.free(new_params);

        i = 0;
        while (i < new_params.len) : (i += 1) {
            new_params[i] = try reader.readInt(u64, .little);
        }
        const new_seed = try reader.readInt(u64, .little);

        self.allocator.free(self.ngram_weights);
        self.allocator.free(self.lsh_hash_params);
        self.ngram_weights = new_weights;
        self.lsh_hash_params = new_params;
        self.num_ngrams = num_ng_usize;
        self.num_hash_functions = num_h_usize;
        self.seed = new_seed;
    }
};

test "Ranker score" {
    const testing = std.testing;
    const gpa = std.testing.allocator;
    var ranker = try Ranker.init(gpa, 4, 8, 42);
    defer ranker.deinit();
    var ssi = SSI.init(gpa);
    defer ssi.deinit();
    try ssi.addSequence(&.{ 1, 2, 3 }, 0, false);
    const score = try ranker.scoreSequence(&.{ 1, 2 }, &ssi);
    try testing.expect(score >= 0.0);
}

test "MinHash signature deterministic" {
    const testing = std.testing;
    const gpa = std.testing.allocator;
    var ranker = try Ranker.init(gpa, 1, 32, 42);
    defer ranker.deinit();
    const sig1 = try ranker.minHashSignature(&.{ 1, 2, 3 });
    defer gpa.free(sig1);
    const sig2 = try ranker.minHashSignature(&.{ 1, 2, 3 });
    defer gpa.free(sig2);
    try testing.expectEqualSlices(u64, sig1, sig2);
}

test "Jaccard similarity from signatures" {
    const testing = std.testing;
    const gpa = std.testing.allocator;
    var ranker = try Ranker.init(gpa, 1, 32, 42);
    defer ranker.deinit();
    const sig1 = try ranker.minHashSignature(&.{ 1, 2, 3 });
    defer gpa.free(sig1);
    const sig2 = try ranker.minHashSignature(&.{ 1, 2, 3 });
    defer gpa.free(sig2);
    const sim = Ranker.jaccardSimilarityFromSignatures(sig1, sig2);
    try testing.expectApproxEqAbs(@as(f32, 1.0), sim, @as(f32, 0.01));
}

test "Token diversity" {
    const testing = std.testing;
    const gpa = std.testing.allocator;
    var ranker = try Ranker.init(gpa, 1, 1, 42);
    defer ranker.deinit();
    const div1 = try ranker.computeTokenDiversity(&.{ 1, 1, 1, 1 }, gpa);
    const div2 = try ranker.computeTokenDiversity(&.{ 1, 2, 3, 4 }, gpa);
    try testing.expect(div2 > div1);
}

test "Token overlap" {
    const testing = std.testing;
    const gpa = std.testing.allocator;
    var ranker = try Ranker.init(gpa, 1, 1, 42);
    defer ranker.deinit();
    const overlap = try ranker.computeTokenOverlap(&.{ 1, 2, 3 }, &.{ 2, 3, 4 }, gpa);
    try testing.expect(overlap > 0.0 and overlap <= 1.0);
}

test "Estimate Jaccard" {
    const testing = std.testing;
    const gpa = std.testing.allocator;
    var set1 = try BitSet.init(gpa, 128);
    defer set1.deinit();
    set1.set(0);
    set1.set(64);
    var set2 = try BitSet.init(gpa, 128);
    defer set2.deinit();
    set2.set(0);
    const est = Ranker.estimateJaccard(set1, set2);
    try testing.expect(est >= 0.0 and est <= 1.0);
}

test "Estimate Jaccard empty sets" {
    const testing = std.testing;
    const gpa = std.testing.allocator;
    var set1 = try BitSet.init(gpa, 64);
    defer set1.deinit();
    var set2 = try BitSet.init(gpa, 64);
    defer set2.deinit();
    const est = Ranker.estimateJaccard(set1, set2);
    try testing.expectApproxEqAbs(@as(f32, 1.0), est, 0.01);
}

test "Vector cosine score" {
    const testing = std.testing;
    const gpa = std.testing.allocator;
    var emb = try Tensor.init(gpa, &.{3});
    defer emb.deinit();
    emb.data[0] = 1.0;
    emb.data[1] = 0.0;
    emb.data[2] = 0.0;
    var qemb = try Tensor.init(gpa, &.{3});
    defer qemb.deinit();
    qemb.data[0] = 1.0;
    qemb.data[1] = 0.0;
    qemb.data[2] = 0.0;
    const score = try Ranker.vectorScore(&emb, &qemb);
    try testing.expectApproxEqAbs(@as(f32, 1.0), score, @as(f32, 0.01));
}

test "Dot product score" {
    const testing = std.testing;
    const gpa = std.testing.allocator;
    var emb = try Tensor.init(gpa, &.{3});
    defer emb.deinit();
    emb.data[0] = 1.0;
    emb.data[1] = 2.0;
    emb.data[2] = 3.0;
    var qemb = try Tensor.init(gpa, &.{3});
    defer qemb.deinit();
    qemb.data[0] = 1.0;
    qemb.data[1] = 2.0;
    qemb.data[2] = 3.0;
    const score = try Ranker.dotProductScore(&emb, &qemb);
    try testing.expectApproxEqAbs(@as(f32, 14.0), score, @as(f32, 0.01));
}

test "Weighted average" {
    const testing = std.testing;
    const scores = [_]f32{ 0.5, 0.8, 0.3 };
    const weights = [_]f32{ 1.0, 2.0, 1.0 };
    const avg = try Ranker.weightedAverage(&scores, &weights);
    try testing.expect(avg > 0.0 and avg < 1.0);
}

test "Weighted average rejects negative weights" {
    const testing = std.testing;
    const scores = [_]f32{ 0.5, 0.8 };
    const weights = [_]f32{ 1.0, -0.5 };
    try testing.expectError(error.InvalidParameter, Ranker.weightedAverage(&scores, &weights));
}

test "Exponential decay" {
    const testing = std.testing;
    var scores = [_]f32{ 1.0, 1.0, 1.0, 1.0 };
    try Ranker.exponentialDecay(&scores, 0.9);
    try testing.expect(scores[0] > scores[1]);
    try testing.expect(scores[1] > scores[2]);
    try testing.expect(scores[2] > scores[3]);
}

test "Normalize scores" {
    const testing = std.testing;
    var scores = [_]f32{ 10.0, 20.0, 30.0, 40.0 };
    Ranker.normalizeScoresStatic(&scores);
    try testing.expectApproxEqAbs(@as(f32, 0.0), scores[0], @as(f32, 0.01));
    try testing.expectApproxEqAbs(@as(f32, 1.0), scores[3], @as(f32, 0.01));
}

test "MinHash bitmask signature identical sequences agree" {
    const testing = std.testing;
    const gpa = std.testing.allocator;
    var ranker = try Ranker.init(gpa, 1, 128, 42);
    defer ranker.deinit();
    const mask1 = try ranker.minHashBitmask(&.{ 5, 6, 7, 8 });
    defer gpa.free(mask1);
    const mask2 = try ranker.minHashBitmask(&.{ 5, 6, 7, 8 });
    defer gpa.free(mask2);
    const sim = Ranker.jaccardFromBitmasks(mask1, mask2, 128);
    try testing.expectApproxEqAbs(@as(f32, 1.0), sim, @as(f32, 0.01));
}

test "MinHash bitmask signature disjoint sequences below one" {
    const testing = std.testing;
    const gpa = std.testing.allocator;
    var ranker = try Ranker.init(gpa, 1, 128, 42);
    defer ranker.deinit();
    const mask1 = try ranker.minHashBitmask(&.{ 1, 2, 3, 4 });
    defer gpa.free(mask1);
    const mask2 = try ranker.minHashBitmask(&.{ 100, 200, 300, 400 });
    defer gpa.free(mask2);
    const sim = Ranker.jaccardFromBitmasks(mask1, mask2, 128);
    try testing.expect(sim < 1.0);
    try testing.expect(sim >= 0.0);
}

test "Jaccard signature bitmask correlates with true overlap" {
    const testing = std.testing;
    const gpa = std.testing.allocator;
    var ranker = try Ranker.init(gpa, 1, 256, 7);
    defer ranker.deinit();
    const sim_high = try ranker.jaccardSignatureBitmask(&.{ 1, 2, 3, 4, 5 }, &.{ 1, 2, 3, 4, 5, 6 });
    const sim_low = try ranker.jaccardSignatureBitmask(&.{ 1, 2, 3, 4, 5 }, &.{ 50, 60, 70, 80, 90 });
    try testing.expect(sim_high > sim_low);
}

test "Content scoring reflects indexed similarity" {
    const testing = std.testing;
    const gpa = std.testing.allocator;
    var ranker = try Ranker.init(gpa, 4, 8, 42);
    defer ranker.deinit();
    var ssi = SSI.init(gpa);
    defer ssi.deinit();
    try ssi.addSequence(&.{ 1, 2, 3, 4, 5 }, 0, true);
    const matching = try ranker.scoreSequence(&.{ 1, 2, 3 }, &ssi);
    const unrelated = try ranker.scoreSequence(&.{ 900, 901, 902, 903, 904 }, &ssi);
    try testing.expect(matching > unrelated);
    try testing.expect(matching > 0.001);
}

test "Ngram order above limit rejected" {
    const testing = std.testing;
    const gpa = std.testing.allocator;
    try testing.expectError(error.InvalidParameter, Ranker.init(gpa, 65, 8, 42));
}

test "Model roundtrip preserves weights and params" {
    const testing = std.testing;
    const gpa = std.testing.allocator;
    var ranker = try Ranker.init(gpa, 4, 8, 42);
    defer ranker.deinit();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const path = try tmp.dir.realpathAlloc(gpa, ".");
    defer gpa.free(path);
    const full = try std.fs.path.join(gpa, &.{ path, "m.bin" });
    defer gpa.free(full);
    try ranker.exportModel(full);

    var r2 = try Ranker.init(gpa, 1, 1, 1);
    defer r2.deinit();
    try r2.importModel(full);
    try testing.expectEqual(@as(usize, 4), r2.num_ngrams);
    try testing.expectEqual(@as(usize, 8), r2.num_hash_functions);
    try testing.expectEqual(@as(usize, 4), r2.ngram_weights.len);
    try testing.expectEqual(@as(usize, 16), r2.lsh_hash_params.len);
    try testing.expectEqual(@as(u64, 42), r2.seed);
    try testing.expectEqualSlices(f32, ranker.ngram_weights, r2.ngram_weights);
}

test "ImportModel truncated file errors and preserves state" {
    const testing = std.testing;
    const gpa = std.testing.allocator;
    var r = try Ranker.init(gpa, 2, 4, 99);
    defer r.deinit();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const path = try tmp.dir.realpathAlloc(gpa, ".");
    defer gpa.free(path);
    const full = try std.fs.path.join(gpa, &.{ path, "trunc.bin" });
    defer gpa.free(full);
    var bytes = std.ArrayList(u8).init(gpa);
    defer bytes.deinit();
    bytes.appendSlice(&[_]u8{2}) catch unreachable;
    bytes.appendSlice(&mem.toBytes(mem.nativeToLittle(u64, 3))) catch unreachable;
    bytes.appendSlice(&mem.toBytes(mem.nativeToLittle(u64, 3))) catch unreachable;
    const w0: u32 = @bitCast(@as(f32, 0.25));
    const w1: u32 = @bitCast(@as(f32, 0.50));
    const w2: u32 = @bitCast(@as(f32, 0.75));
    bytes.appendSlice(&mem.toBytes(mem.nativeToLittle(u32, w0))) catch unreachable;
    bytes.appendSlice(&mem.toBytes(mem.nativeToLittle(u32, w1))) catch unreachable;
    bytes.appendSlice(&mem.toBytes(mem.nativeToLittle(u32, w2))) catch unreachable;
    try tmp.dir.writeFile(.{ .sub_path = "trunc.bin", .data = bytes.items });
    try testing.expectError(error.EndOfStream, r.importModel(full));
    try testing.expectEqual(@as(usize, 2), r.num_ngrams);
    try testing.expectEqual(@as(usize, 4), r.num_hash_functions);
    try testing.expectEqual(@as(usize, 2), r.ngram_weights.len);
    try testing.expectEqual(@as(usize, 8), r.lsh_hash_params.len);
    try testing.expectEqual(@as(u64, 99), r.seed);
}

test "CalibrateWeights leaves weights unchanged when labels match predictions" {
    const testing = std.testing;
    const gpa = std.testing.allocator;
    var ranker = try Ranker.init(gpa, 2, 8, 42);
    defer ranker.deinit();
    var ssi = SSI.init(gpa);
    defer ssi.deinit();
    try ssi.addSequence(&.{ 1, 2, 3, 4, 5 }, 0, true);
    const w0_before = ranker.ngram_weights[0];
    const w1_before = ranker.ngram_weights[1];
    const pred = try ranker.scoreSequence(&.{ 1, 2, 3 }, &ssi);
    const data = [_][]const u32{ &.{ 1, 2, 3 } };
    const labels = [_]f32{pred};
    try ranker.calibrateWeights(&data, &labels, &ssi, 1);
    try testing.expectApproxEqAbs(w0_before, ranker.ngram_weights[0], 1e-4);
    try testing.expectApproxEqAbs(w1_before, ranker.ngram_weights[1], 1e-4);
}

test "TopKHeap returns at most k results" {
    const testing = std.testing;
    const gpa = std.testing.allocator;
    var ranker = try Ranker.init(gpa, 4, 8, 42);
    defer ranker.deinit();
    var ssi = SSI.init(gpa);
    defer ssi.deinit();
    const out = try ranker.topKHeap(&ssi, &.{1}, 3, gpa);
    defer {
        for (out) |*rs| rs.deinit(gpa);
        gpa.free(out);
    }
    try testing.expect(out.len <= 3);
}

test "StreamingRank returns at most k segments" {
    const testing = std.testing;
    const gpa = std.testing.allocator;
    var ranker = try Ranker.init(gpa, 4, 8, 42);
    defer ranker.deinit();
    var ssi = SSI.init(gpa);
    defer ssi.deinit();
    var bytes = std.ArrayList(u8).init(gpa);
    defer bytes.deinit();
    var i: u32 = 0;
    while (i < 3000) : (i += 1) {
        const le = mem.toBytes(mem.nativeToLittle(u32, i));
        bytes.appendSlice(&le) catch unreachable;
    }
    var reader = std.io.fixedBufferStream(bytes.items);
    const out = try ranker.streamingRank(&reader.reader(), &ssi, 4, gpa);
    defer {
        for (out) |*rs| rs.deinit(gpa);
        gpa.free(out);
    }
    try testing.expect(out.len <= 4);
}

test "RankCandidatesWithQuery sorts by combined score" {
    const testing = std.testing;
    const gpa = std.testing.allocator;
    var ranker = try Ranker.init(gpa, 4, 8, 42);
    defer ranker.deinit();
    var ssi = SSI.init(gpa);
    defer ssi.deinit();
    try ssi.addSequence(&.{ 1, 2, 3, 4, 5 }, 0, true);
    const c1 = try RankedSegment.init(gpa, @constCast(&[_]u32{ 1, 2, 3 }), 0.0, 0, true);
    const c2 = try RankedSegment.init(gpa, @constCast(&[_]u32{ 900, 901, 902 }), 0.0, 1, false);
    var cands = [_]RankedSegment{ c1, c2 };
    defer {
        for (&cands) |*c| c.deinit(gpa);
    }
    try ranker.rankCandidatesWithQuery(&cands, &.{ 1, 2, 3 }, &ssi, gpa);
    try testing.expect(cands[0].score >= cands[1].score);
}



fn clamp01(x: f64) f64 {
    if (x < 0.0) return 0.0;
    if (x > 1.0) return 1.0;
    return x;
}

fn hashTripletFields(subject: []const u8, relation: []const u8, object: []const u8, confidence: f64, extraction_time: i128) [32]u8 {
    var h = Sha256.init(.{});
    h.update(subject);
    h.update(&[_]u8{0});
    h.update(relation);
    h.update(&[_]u8{0});
    h.update(object);
    h.update(&[_]u8{0});
    const conf_bits: u64 = @bitCast(confidence);
    var conf_le: [8]u8 = undefined;
    std.mem.writeInt(u64, &conf_le, conf_bits, .little);
    h.update(&conf_le);
    var time_le: [16]u8 = undefined;
    std.mem.writeInt(i128, &time_le, extraction_time, .little);
    h.update(&time_le);
    var out: [32]u8 = undefined;
    h.final(&out);
    return out;
}

fn hashTripletIdentity(subject: []const u8, relation: []const u8, object: []const u8) [32]u8 {
    var h = Sha256.init(.{});
    h.update(subject);
    h.update(&[_]u8{0});
    h.update(relation);
    h.update(&[_]u8{0});
    h.update(object);
    var out: [32]u8 = undefined;
    h.final(&out);
    return out;
}

pub const ExtractionStage = enum(u8) {
    tokenization = 0,
    triplet_extraction = 1,
    validation = 2,
    integration = 3,
    indexing = 4,

    pub fn toString(self: ExtractionStage) []const u8 {
        return switch (self) {
            .tokenization => "tokenization",
            .triplet_extraction => "triplet_extraction",
            .validation => "validation",
            .integration => "integration",
            .indexing => "indexing",
        };
    }

    pub fn fromString(s: []const u8) ?ExtractionStage {
        const t = std.mem.trim(u8, s, " \t\r\n");
        if (std.ascii.eqlIgnoreCase(t, "tokenization")) return .tokenization;
        if (std.ascii.eqlIgnoreCase(t, "triplet_extraction")) return .triplet_extraction;
        if (std.ascii.eqlIgnoreCase(t, "validation")) return .validation;
        if (std.ascii.eqlIgnoreCase(t, "integration")) return .integration;
        if (std.ascii.eqlIgnoreCase(t, "indexing")) return .indexing;
        return null;
    }

    pub fn next(self: ExtractionStage) ?ExtractionStage {
        return switch (self) {
            .tokenization => .triplet_extraction,
            .triplet_extraction => .validation,
            .validation => .integration,
            .integration => .indexing,
            .indexing => null,
        };
    }
};

const WordToken = struct {
    text: []const u8,
    start: usize,
    end: usize,
};

const MorphemeMatch = struct {
    match_start: usize,
    match_end: usize,
};

fn tokenizeIntoWords(text: []const u8, tokens: *ArrayList(WordToken)) !void {
    var i: usize = 0;
    while (i < text.len) {
        while (i < text.len and (text[i] == ' ' or text[i] == '\t' or text[i] == '\n' or text[i] == '\r' or text[i] == ',' or text[i] == ';' or text[i] == ':')) {
            i += 1;
        }
        if (i >= text.len) break;
        const start = i;
        while (i < text.len and text[i] != ' ' and text[i] != '\t' and text[i] != '\n' and text[i] != '\r' and text[i] != ',' and text[i] != ';' and text[i] != ':') {
            i += 1;
        }
        if (start < i) {
            try tokens.append(.{ .text = text[start..i], .start = start, .end = i });
        }
    }
}

fn stemWord(word: []const u8) []const u8 {
    if (word.len > 5 and std.mem.endsWith(u8, word, "ting")) {
        return word[0 .. word.len - 4];
    }
    if (word.len > 4 and std.mem.endsWith(u8, word, "ing")) {
        if (word.len > 5 and word[word.len - 4] == word[word.len - 5]) {
            return word[0 .. word.len - 4];
        }
        return word[0 .. word.len - 3];
    }
    if (word.len > 4 and std.mem.endsWith(u8, word, "ated")) {
        return word[0 .. word.len - 4];
    }
    if (word.len > 3 and std.mem.endsWith(u8, word, "ed")) {
        if (word.len > 4 and word[word.len - 3] == word[word.len - 4]) {
            return word[0 .. word.len - 3];
        }
        return word[0 .. word.len - 2];
    }
    if (word.len > 4 and std.mem.endsWith(u8, word, "ies")) {
        return word[0 .. word.len - 3];
    }
    if (word.len > 4 and std.mem.endsWith(u8, word, "ches") or (word.len > 4 and std.mem.endsWith(u8, word, "shes")) or (word.len > 4 and std.mem.endsWith(u8, word, "sses"))) {
        return word[0 .. word.len - 2];
    }
    if (word.len > 3 and std.mem.endsWith(u8, word, "es")) {
        return word[0 .. word.len - 2];
    }
    if (word.len > 2 and std.mem.endsWith(u8, word, "s") and !std.mem.endsWith(u8, word, "ss") and !std.mem.endsWith(u8, word, "us")) {
        return word[0 .. word.len - 1];
    }
    if (word.len > 4 and std.mem.endsWith(u8, word, "ally")) {
        return word[0 .. word.len - 4];
    }
    if (word.len > 3 and std.mem.endsWith(u8, word, "ly")) {
        return word[0 .. word.len - 2];
    }
    if (word.len > 4 and std.mem.endsWith(u8, word, "ment")) {
        return word[0 .. word.len - 4];
    }
    if (word.len > 4 and std.mem.endsWith(u8, word, "ness")) {
        return word[0 .. word.len - 4];
    }
    if (word.len > 3 and std.mem.endsWith(u8, word, "er") and !std.mem.endsWith(u8, word, "ver") and !std.mem.endsWith(u8, word, "her")) {
        if (word.len > 4 and word[word.len - 3] == word[word.len - 4]) {
            return word[0 .. word.len - 3];
        }
        return word[0 .. word.len - 2];
    }
    if (word.len > 4 and std.mem.endsWith(u8, word, "est")) {
        return word[0 .. word.len - 3];
    }
    return word;
}

fn wordsMatchStem(a: []const u8, b: []const u8) bool {
    if (std.mem.eql(u8, a, b)) return true;
    var a_buf: [256]u8 = undefined;
    var b_buf: [256]u8 = undefined;
    if (a.len >= a_buf.len or b.len >= b_buf.len) return std.mem.eql(u8, a, b);
    const a_lower = std.ascii.lowerString(&a_buf, a);
    const b_lower = std.ascii.lowerString(&b_buf, b);
    if (std.mem.eql(u8, a_lower, b_lower)) return true;
    const a_stem = stemWord(a_lower);
    const b_stem = stemWord(b_lower);
    if (a_stem.len > 0 and b_stem.len > 0 and std.mem.eql(u8, a_stem, b_stem)) return true;
    return false;
}

fn matchPatternMorphemeAware(sentence: []const u8, pattern: []const u8, allocator: Allocator) ?MorphemeMatch {
    var sent_tokens = ArrayList(WordToken).init(allocator);
    defer sent_tokens.deinit();
    tokenizeIntoWords(sentence, &sent_tokens) catch return null;

    var pat_tokens = ArrayList(WordToken).init(allocator);
    defer pat_tokens.deinit();
    tokenizeIntoWords(pattern, &pat_tokens) catch return null;

    if (pat_tokens.items.len == 0 or sent_tokens.items.len == 0) return null;
    if (pat_tokens.items.len > sent_tokens.items.len) return null;

    var si: usize = 0;
    while (si <= sent_tokens.items.len - pat_tokens.items.len) {
        var all_match = true;
        for (pat_tokens.items, 0..) |pat_tok, pi| {
            if (!wordsMatchStem(sent_tokens.items[si + pi].text, pat_tok.text)) {
                all_match = false;
                break;
            }
        }
        if (all_match) {
            const last_idx = si + pat_tokens.items.len - 1;
            return MorphemeMatch{
                .match_start = sent_tokens.items[si].start,
                .match_end = sent_tokens.items[last_idx].end,
            };
        }
        si += 1;
    }
    return null;
}

pub const RelationalTriplet = struct {
    subject: []u8,
    relation: []u8,
    object: []u8,
    confidence: f64,
    source_hash: [32]u8,
    extraction_time: i128,
    allocator: Allocator,
    metadata: StringHashMap([]u8),

    pub fn init(
        allocator: Allocator,
        subject: []const u8,
        relation: []const u8,
        object: []const u8,
        confidence_in: f64,
    ) !RelationalTriplet {
        const now_ns = @as(i64, @truncate(std.time.nanoTimestamp()));

        const s = try allocator.dupe(u8, subject);
        errdefer allocator.free(s);
        const r = try allocator.dupe(u8, relation);
        errdefer allocator.free(r);
        const o = try allocator.dupe(u8, object);
        errdefer allocator.free(o);

        return RelationalTriplet{
            .subject = s,
            .relation = r,
            .object = o,
            .confidence = clamp01(confidence_in),
            .source_hash = hashTripletIdentity(subject, relation, object),
            .extraction_time = now_ns,
            .allocator = allocator,
            .metadata = StringHashMap([]u8).init(allocator),
        };
    }

    pub fn initWithHash(
        allocator: Allocator,
        subject: []const u8,
        relation: []const u8,
        object: []const u8,
        confidence_in: f64,
        source_hash: [32]u8,
        extraction_time: i128,
    ) !RelationalTriplet {
        const s = try allocator.dupe(u8, subject);
        errdefer allocator.free(s);
        const r = try allocator.dupe(u8, relation);
        errdefer allocator.free(r);
        const o = try allocator.dupe(u8, object);
        errdefer allocator.free(o);

        return RelationalTriplet{
            .subject = s,
            .relation = r,
            .object = o,
            .confidence = clamp01(confidence_in),
            .source_hash = source_hash,
            .extraction_time = extraction_time,
            .allocator = allocator,
            .metadata = StringHashMap([]u8).init(allocator),
        };
    }

    pub fn deinit(self: *RelationalTriplet) void {
        self.allocator.free(self.subject);
        self.allocator.free(self.relation);
        self.allocator.free(self.object);

        var it = self.metadata.iterator();
        while (it.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            self.allocator.free(entry.value_ptr.*);
        }
        self.metadata.deinit();

        self.subject = &[_]u8{};
        self.relation = &[_]u8{};
        self.object = &[_]u8{};
    }

    pub fn clone(self: *const RelationalTriplet, allocator: Allocator) !RelationalTriplet {
        const s = try allocator.dupe(u8, self.subject);
        errdefer allocator.free(s);
        const r = try allocator.dupe(u8, self.relation);
        errdefer allocator.free(r);
        const o = try allocator.dupe(u8, self.object);
        errdefer allocator.free(o);

        var t = RelationalTriplet{
            .subject = s,
            .relation = r,
            .object = o,
            .confidence = self.confidence,
            .source_hash = self.source_hash,
            .extraction_time = self.extraction_time,
            .allocator = allocator,
            .metadata = StringHashMap([]u8).init(allocator),
        };
        errdefer t.deinit();

        var it = self.metadata.iterator();
        while (it.next()) |entry| {
            const k = try allocator.dupe(u8, entry.key_ptr.*);
            errdefer allocator.free(k);
            const v = try allocator.dupe(u8, entry.value_ptr.*);
            errdefer allocator.free(v);
            try t.metadata.put(k, v);
        }
        return t;
    }

    pub fn computeHash(self: *const RelationalTriplet) [32]u8 {
        return hashTripletFields(self.subject, self.relation, self.object, self.confidence, self.extraction_time);
    }

    pub fn setMetadata(self: *RelationalTriplet, key: []const u8, value: []const u8) !void {
        const v_copy = try self.allocator.dupe(u8, value);
        errdefer self.allocator.free(v_copy);

        if (self.metadata.getPtr(key)) |existing_v| {
            self.allocator.free(existing_v.*);
            existing_v.* = v_copy;
            return;
        }

        const k_copy = try self.allocator.dupe(u8, key);
        errdefer self.allocator.free(k_copy);

        try self.metadata.put(k_copy, v_copy);
    }

    pub fn getMetadata(self: *const RelationalTriplet, key: []const u8) ?[]const u8 {
        if (self.metadata.get(key)) |v| return v;
        return null;
    }

    pub fn equals(self: *const RelationalTriplet, other: *const RelationalTriplet) bool {
        return std.mem.eql(u8, self.subject, other.subject) and
            std.mem.eql(u8, self.relation, other.relation) and
            std.mem.eql(u8, self.object, other.object);
    }

    pub fn hashEquals(self: *const RelationalTriplet, other: *const RelationalTriplet) bool {
        return std.mem.eql(u8, self.source_hash[0..], other.source_hash[0..]);
    }

    pub fn toGraphElements(self: *const RelationalTriplet, allocator: Allocator) !struct {
        subject_node: Node,
        object_node: Node,
        edge: Edge,
    } {
        var subject_id_hash: [32]u8 = undefined;
        Sha256.hash(self.subject, &subject_id_hash, .{});
        var subject_id: [16]u8 = undefined;
        @memcpy(subject_id[0..], subject_id_hash[0..16]);

        var object_id_hash: [32]u8 = undefined;
        Sha256.hash(self.object, &object_id_hash, .{});
        var object_id: [16]u8 = undefined;
        @memcpy(object_id[0..], object_id_hash[0..16]);

        var subject_id_str: [32]u8 = undefined;
        _ = try std.fmt.bufPrint(subject_id_str[0..], "{s}", .{std.fmt.fmtSliceHexLower(subject_id[0..])});

        var object_id_str: [32]u8 = undefined;
        _ = try std.fmt.bufPrint(object_id_str[0..], "{s}", .{std.fmt.fmtSliceHexLower(object_id[0..])});

        const c = clamp01(self.confidence);
        const imag_sq = 1.0 - c * c;
        const imag = @sqrt(@max(0.0, imag_sq));
        const quantum_state = Complex(f64).init(c, imag);

        const period_ns: i128 = 360 * 1_000_000_000;
        const mod_ns: i128 = @mod(self.extraction_time, period_ns);
        const phase = @as(f64, @floatFromInt(mod_ns)) / @as(f64, @floatFromInt(period_ns)) * std.math.pi * 2.0;

        var subject_node = try Node.initWithComplex(
            allocator,
            subject_id_str[0..],
            self.subject,
            quantum_state,
            phase,
        );
        errdefer subject_node.deinit();
        try subject_node.setMetadata("type", "entity");
        try subject_node.setMetadata("role", "subject");

        var object_node = try Node.initWithComplex(
            allocator,
            object_id_str[0..],
            self.object,
            quantum_state,
            phase,
        );
        errdefer object_node.deinit();
        try object_node.setMetadata("type", "entity");
        try object_node.setMetadata("role", "object");

        var edge = try Edge.initWithComplex(
            allocator,
            subject_id_str[0..],
            object_id_str[0..],
            .coherent,
            c,
            quantum_state,
            1.0,
        );
        errdefer edge.deinit();
        try edge.setMetadata("relation", self.relation);

        var conf_buf: [64]u8 = undefined;
        const conf_str = try std.fmt.bufPrint(conf_buf[0..], "{d:.6}", .{c});
        try edge.setMetadata("confidence", conf_str);

        return .{
            .subject_node = subject_node,
            .object_node = object_node,
            .edge = edge,
        };
    }
};

pub const ValidationResult = struct {
    triplet: *RelationalTriplet,
    is_valid: bool,
    confidence_adjusted: f64,
    validation_method: []const u8,
    conflicts: ArrayList(*RelationalTriplet),
    anomaly_score: f64,
    validation_time: i128,
    allocator: Allocator,

    pub fn init(allocator: Allocator, triplet: *RelationalTriplet) ValidationResult {
        return ValidationResult{
            .triplet = triplet,
            .is_valid = true,
            .confidence_adjusted = triplet.confidence,
            .validation_method = "",
            .conflicts = ArrayList(*RelationalTriplet).init(allocator),
            .anomaly_score = 0.0,
            .validation_time = @as(i64, @truncate(std.time.nanoTimestamp())),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *ValidationResult) void {
        self.conflicts.deinit();
    }

    pub fn addConflict(self: *ValidationResult, conflict: *RelationalTriplet) !void {
        try self.conflicts.append(conflict);
    }

    pub fn hasConflicts(self: *const ValidationResult) bool {
        return self.conflicts.items.len > 0;
    }

    pub fn conflictCount(self: *const ValidationResult) usize {
        return self.conflicts.items.len;
    }

    pub fn setValidationMethod(self: *ValidationResult, method: []const u8) void {
        self.validation_method = method;
    }
};

pub const KnowledgeGraphIndex = struct {
    subject_index: StringHashMap(ArrayList(*RelationalTriplet)),
    relation_index: StringHashMap(ArrayList(*RelationalTriplet)),
    object_index: StringHashMap(ArrayList(*RelationalTriplet)),
    all_triplets: ArrayList(*RelationalTriplet),
    allocator: Allocator,

    pub fn init(allocator: Allocator) KnowledgeGraphIndex {
        return KnowledgeGraphIndex{
            .subject_index = StringHashMap(ArrayList(*RelationalTriplet)).init(allocator),
            .relation_index = StringHashMap(ArrayList(*RelationalTriplet)).init(allocator),
            .object_index = StringHashMap(ArrayList(*RelationalTriplet)).init(allocator),
            .all_triplets = ArrayList(*RelationalTriplet).init(allocator),
            .allocator = allocator,
        };
    }

    fn deinitIndexMap(self: *KnowledgeGraphIndex, map: *StringHashMap(ArrayList(*RelationalTriplet))) void {
        var it = map.iterator();
        while (it.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            entry.value_ptr.*.deinit();
        }
        map.deinit();
    }

    pub fn deinit(self: *KnowledgeGraphIndex) void {
        self.deinitIndexMap(&self.subject_index);
        self.deinitIndexMap(&self.relation_index);
        self.deinitIndexMap(&self.object_index);

        for (self.all_triplets.items) |triplet| {
            triplet.deinit();
            self.allocator.destroy(triplet);
        }
        self.all_triplets.deinit();
    }

    fn indexIntoMap(
        self: *KnowledgeGraphIndex,
        map: *StringHashMap(ArrayList(*RelationalTriplet)),
        key: []const u8,
        triplet: *RelationalTriplet,
    ) !void {
        const gop = try map.getOrPut(key);
        if (!gop.found_existing) {
            const key_copy = try self.allocator.dupe(u8, key);
            errdefer self.allocator.free(key_copy);
            gop.key_ptr.* = key_copy;
            gop.value_ptr.* = ArrayList(*RelationalTriplet).init(self.allocator);
        }
        try gop.value_ptr.*.append(triplet);
    }

    pub fn index(self: *KnowledgeGraphIndex, triplet: *RelationalTriplet) !void {
        try self.all_triplets.append(triplet);
        errdefer {
            _ = self.all_triplets.pop();
        }

        try self.indexIntoMap(&self.subject_index, triplet.subject, triplet);
        try self.indexIntoMap(&self.relation_index, triplet.relation, triplet);
        try self.indexIntoMap(&self.object_index, triplet.object, triplet);
    }

    pub fn query(
        self: *KnowledgeGraphIndex,
        subject: ?[]const u8,
        relation: ?[]const u8,
        object: ?[]const u8,
        allocator: Allocator,
    ) !ArrayList(*RelationalTriplet) {
        var results = ArrayList(*RelationalTriplet).init(allocator);

        if (subject == null and relation == null and object == null) {
            try results.appendSlice(self.all_triplets.items);
            return results;
        }

        var best: ?*ArrayList(*RelationalTriplet) = null;
        var best_len: usize = std.math.maxInt(usize);

        if (subject) |s| {
            if (self.subject_index.getPtr(s)) |list| {
                if (list.items.len < best_len) {
                    best = list;
                    best_len = list.items.len;
                }
            } else {
                return results;
            }
        }

        if (relation) |r| {
            if (self.relation_index.getPtr(r)) |list| {
                if (list.items.len < best_len) {
                    best = list;
                    best_len = list.items.len;
                }
            } else {
                return results;
            }
        }

        if (object) |o| {
            if (self.object_index.getPtr(o)) |list| {
                if (list.items.len < best_len) {
                    best = list;
                    best_len = list.items.len;
                }
            } else {
                return results;
            }
        }

        const cands = best orelse return results;

        for (cands.items) |t| {
            if (subject) |s| {
                if (!std.mem.eql(u8, t.subject, s)) continue;
            }
            if (relation) |r| {
                if (!std.mem.eql(u8, t.relation, r)) continue;
            }
            if (object) |o| {
                if (!std.mem.eql(u8, t.object, o)) continue;
            }
            try results.append(t);
        }

        return results;
    }

    pub fn queryMorphemeAware(
        self: *KnowledgeGraphIndex,
        subject: ?[]const u8,
        relation: ?[]const u8,
        object: ?[]const u8,
        allocator: Allocator,
    ) !ArrayList(*RelationalTriplet) {
        var results = ArrayList(*RelationalTriplet).init(allocator);

        for (self.all_triplets.items) |t| {
            var subj_match = true;
            var rel_match = true;
            var obj_match = true;

            if (subject) |s| {
                var s_buf: [256]u8 = undefined;
                var t_buf: [256]u8 = undefined;
                if (s.len < s_buf.len and t.subject.len < t_buf.len) {
                    const s_lower = std.ascii.lowerString(&s_buf, s);
                    const t_lower = std.ascii.lowerString(&t_buf, t.subject);
                    const s_stem = stemWord(s_lower);
                    const t_stem = stemWord(t_lower);
                    subj_match = std.mem.eql(u8, s_stem, t_stem) or std.mem.eql(u8, s, t.subject);
                } else {
                    subj_match = std.mem.eql(u8, s, t.subject);
                }
            }
            if (relation) |r| {
                var r_buf: [256]u8 = undefined;
                var t_buf: [256]u8 = undefined;
                if (r.len < r_buf.len and t.relation.len < t_buf.len) {
                    const r_lower = std.ascii.lowerString(&r_buf, r);
                    const t_lower = std.ascii.lowerString(&t_buf, t.relation);
                    const r_stem = stemWord(r_lower);
                    const t_stem = stemWord(t_lower);
                    rel_match = std.mem.eql(u8, r_stem, t_stem) or std.mem.eql(u8, r, t.relation);
                } else {
                    rel_match = std.mem.eql(u8, r, t.relation);
                }
            }
            if (object) |o| {
                var o_buf: [256]u8 = undefined;
                var t_buf: [256]u8 = undefined;
                if (o.len < o_buf.len and t.object.len < t_buf.len) {
                    const o_lower = std.ascii.lowerString(&o_buf, o);
                    const t_lower = std.ascii.lowerString(&t_buf, t.object);
                    const o_stem = stemWord(o_lower);
                    const t_stem = stemWord(t_lower);
                    obj_match = std.mem.eql(u8, o_stem, t_stem) or std.mem.eql(u8, o, t.object);
                } else {
                    obj_match = std.mem.eql(u8, o, t.object);
                }
            }

            if (subj_match and rel_match and obj_match) {
                try results.append(t);
            }
        }

        return results;
    }

    pub fn queryBySubject(self: *KnowledgeGraphIndex, subject: []const u8) []*RelationalTriplet {
        if (self.subject_index.getPtr(subject)) |list| return list.items;
        return &[_]*RelationalTriplet{};
    }

    pub fn queryByRelation(self: *KnowledgeGraphIndex, relation: []const u8) []*RelationalTriplet {
        if (self.relation_index.getPtr(relation)) |list| return list.items;
        return &[_]*RelationalTriplet{};
    }

    pub fn queryByObject(self: *KnowledgeGraphIndex, object: []const u8) []*RelationalTriplet {
        if (self.object_index.getPtr(object)) |list| return list.items;
        return &[_]*RelationalTriplet{};
    }

    fn removeFromList(list: *ArrayList(*RelationalTriplet), triplet: *RelationalTriplet) bool {
        var removed_any = false;
        var i: usize = 0;
        while (i < list.items.len) {
            if (list.items[i] == triplet) {
                _ = list.orderedRemove(i);
                removed_any = true;
            } else {
                i += 1;
            }
        }
        return removed_any;
    }

    fn removeFromMap(
        self: *KnowledgeGraphIndex,
        map: *StringHashMap(ArrayList(*RelationalTriplet)),
        key: []const u8,
        triplet: *RelationalTriplet,
    ) bool {
        if (map.getPtr(key)) |list| {
            const removed = removeFromList(list, triplet);
            if (removed and list.items.len == 0) {
                if (map.fetchRemove(key)) |kv| {
                    self.allocator.free(kv.key);
                    kv.value.deinit();
                }
            }
            return removed;
        }
        return false;
    }

    pub fn remove(self: *KnowledgeGraphIndex, triplet: *RelationalTriplet) bool {
        var removed_any = false;

        removed_any = self.removeFromMap(&self.subject_index, triplet.subject, triplet) or removed_any;
        removed_any = self.removeFromMap(&self.relation_index, triplet.relation, triplet) or removed_any;
        removed_any = self.removeFromMap(&self.object_index, triplet.object, triplet) or removed_any;

        var i: usize = 0;
        while (i < self.all_triplets.items.len) {
            if (self.all_triplets.items[i] == triplet) {
                _ = self.all_triplets.orderedRemove(i);
                removed_any = true;
            } else {
                i += 1;
            }
        }

        return removed_any;
    }

    pub fn count(self: *const KnowledgeGraphIndex) usize {
        return self.all_triplets.items.len;
    }

    pub fn getUniqueSubjects(self: *const KnowledgeGraphIndex) usize {
        return self.subject_index.count();
    }

    pub fn getUniqueRelations(self: *const KnowledgeGraphIndex) usize {
        return self.relation_index.count();
    }

    pub fn getUniqueObjects(self: *const KnowledgeGraphIndex) usize {
        return self.object_index.count();
    }
};

pub const StreamBuffer = struct {
    buffer: []?*RelationalTriplet,
    capacity: usize,
    head: usize,
    tail: usize,
    size: usize,
    allocator: Allocator,
    overflow_count: usize,
    total_pushed: usize,
    total_popped: usize,

    pub fn init(allocator: Allocator, capacity: usize) !StreamBuffer {
        const buf = try allocator.alloc(?*RelationalTriplet, capacity);
        @memset(buf, null);
        return StreamBuffer{
            .buffer = buf,
            .capacity = capacity,
            .head = 0,
            .tail = 0,
            .size = 0,
            .allocator = allocator,
            .overflow_count = 0,
            .total_pushed = 0,
            .total_popped = 0,
        };
    }

    pub fn deinit(self: *StreamBuffer) void {
        self.allocator.free(self.buffer);
        self.buffer = &[_]?*RelationalTriplet{};
        self.capacity = 0;
        self.head = 0;
        self.tail = 0;
        self.size = 0;
    }

    pub fn push(self: *StreamBuffer, triplet: *RelationalTriplet) bool {
        if (self.capacity == 0) {
            self.overflow_count += 1;
            return false;
        }
        if (self.isFull()) {
            self.overflow_count += 1;
            return false;
        }
        self.buffer[self.tail] = triplet;
        self.tail = (self.tail + 1) % self.capacity;
        self.size += 1;
        self.total_pushed += 1;
        return true;
    }

    pub fn pop(self: *StreamBuffer) ?*RelationalTriplet {
        if (self.isEmpty()) return null;
        const t = self.buffer[self.head];
        self.buffer[self.head] = null;
        self.head = (self.head + 1) % self.capacity;
        self.size -= 1;
        self.total_popped += 1;
        return t;
    }

    pub fn peek(self: *const StreamBuffer) ?*RelationalTriplet {
        if (self.isEmpty()) return null;
        return self.buffer[self.head];
    }

    pub fn peekAt(self: *const StreamBuffer, offset: usize) ?*RelationalTriplet {
        if (offset >= self.size) return null;
        if (self.capacity == 0) return null;
        const idx = (self.head + offset) % self.capacity;
        return self.buffer[idx];
    }

    pub fn isFull(self: *const StreamBuffer) bool {
        return self.capacity != 0 and self.size >= self.capacity;
    }

    pub fn isEmpty(self: *const StreamBuffer) bool {
        return self.size == 0;
    }

    pub fn getSize(self: *const StreamBuffer) usize {
        return self.size;
    }

    pub fn getCapacity(self: *const StreamBuffer) usize {
        return self.capacity;
    }

    pub fn clear(self: *StreamBuffer) void {
        if (self.capacity != 0) {
            @memset(self.buffer, null);
        }
        self.head = 0;
        self.tail = 0;
        self.size = 0;
    }

    pub fn getUtilization(self: *const StreamBuffer) f64 {
        if (self.capacity == 0) return 0.0;
        return @as(f64, @floatFromInt(self.size)) / @as(f64, @floatFromInt(self.capacity));
    }
};

pub const PipelineResult = struct {
    triplets_extracted: usize,
    triplets_validated: usize,
    triplets_integrated: usize,
    conflicts_resolved: usize,
    processing_time_ns: i128,
    stage: ExtractionStage,
    success: bool,
    error_message: ?[]const u8,

    pub fn init() PipelineResult {
        return PipelineResult{
            .triplets_extracted = 0,
            .triplets_validated = 0,
            .triplets_integrated = 0,
            .conflicts_resolved = 0,
            .processing_time_ns = 0,
            .stage = .tokenization,
            .success = true,
            .error_message = null,
        };
    }

    pub fn merge(self: *PipelineResult, other: PipelineResult) void {
        self.triplets_extracted += other.triplets_extracted;
        self.triplets_validated += other.triplets_validated;
        self.triplets_integrated += other.triplets_integrated;
        self.conflicts_resolved += other.conflicts_resolved;
        self.processing_time_ns += other.processing_time_ns;
        self.success = self.success and other.success;
        if (self.error_message == null) self.error_message = other.error_message;
    }
};

pub const PipelineStatistics = struct {
    total_extractions: usize,
    total_validations: usize,
    total_integrations: usize,
    average_confidence: f64,
    conflict_rate: f64,
    throughput: f64,
    buffer_utilization: f64,
    unique_subjects: usize,
    unique_relations: usize,
    unique_objects: usize,
    uptime_ms: i64,

    pub fn init() PipelineStatistics {
        return PipelineStatistics{
            .total_extractions = 0,
            .total_validations = 0,
            .total_integrations = 0,
            .average_confidence = 0.0,
            .conflict_rate = 0.0,
            .throughput = 0.0,
            .buffer_utilization = 0.0,
            .unique_subjects = 0,
            .unique_relations = 0,
            .unique_objects = 0,
            .uptime_ms = 0,
        };
    }
};

pub const RelationPattern = struct {
    pattern: []const u8,
    relation_type: []const u8,
    weight: f64,
};

pub const TokenizerConfig = struct {
    min_entity_length: usize,
    max_entity_length: usize,
    min_confidence_threshold: f64,
    enable_coreference: bool,
    language: []const u8,

    pub fn default() TokenizerConfig {
        return TokenizerConfig{
            .min_entity_length = 2,
            .max_entity_length = 100,
            .min_confidence_threshold = 0.3,
            .enable_coreference = true,
            .language = "en",
        };
    }
};

pub const InferenceHook = struct {
    pre_process: ?*const fn (*anyopaque, []const u8) void,
    post_process: ?*const fn (*anyopaque, *PipelineResult) void,
    pre_query: ?*const fn (*anyopaque, ?[]const u8, ?[]const u8, ?[]const u8) void,
    post_query: ?*const fn (*anyopaque, usize) void,
    context: *anyopaque,
};

pub const CREVPipeline = struct {
    kernel: *ChaosCoreKernel,
    triplet_buffer: StreamBuffer,
    knowledge_index: KnowledgeGraphIndex,
    validation_threshold: f64,
    extraction_count: usize,
    validation_count: usize,
    integration_count: usize,
    conflict_count: usize,
    allocator: Allocator,
    start_time: i128,
    total_confidence_sum: f64,
    relation_patterns: ArrayList(RelationPattern),
    tokenizer_config: TokenizerConfig,
    relation_statistics: StringHashMap(RelationStatistics),
    entity_statistics: StringHashMap(EntityStatistics),
    is_running: bool,
    inference_hooks: ArrayList(InferenceHook),

    pub const RelationStatistics = struct {
        count: usize,
        total_confidence: f64,
        m2: f64,
        avg_confidence: f64,

        pub fn init() RelationStatistics {
            return RelationStatistics{
                .count = 0,
                .total_confidence = 0.0,
                .m2 = 0.0,
                .avg_confidence = 0.0,
            };
        }

        pub fn update(self: *RelationStatistics, confidence_in: f64) void {
            const x = clamp01(confidence_in);
            self.count += 1;
            self.total_confidence += x;
            const delta = x - self.avg_confidence;
            self.avg_confidence += delta / @as(f64, @floatFromInt(self.count));
            const delta2 = x - self.avg_confidence;
            self.m2 += delta * delta2;
        }

        pub fn getVariance(self: *const RelationStatistics) f64 {
            if (self.count < 2) return 0.0;
            const v = self.m2 / @as(f64, @floatFromInt(self.count - 1));
            return @max(0.0, v);
        }

        pub fn getStdDev(self: *const RelationStatistics) f64 {
            return @sqrt(self.getVariance());
        }
    };

    pub const EntityStatistics = struct {
        count: usize,
        as_subject: usize,
        as_object: usize,
        total_confidence: f64,

        pub fn init() EntityStatistics {
            return EntityStatistics{
                .count = 0,
                .as_subject = 0,
                .as_object = 0,
                .total_confidence = 0.0,
            };
        }
    };

    pub fn init(allocator: Allocator, kernel: *ChaosCoreKernel) !CREVPipeline {
        var pipeline = CREVPipeline{
            .kernel = kernel,
            .triplet_buffer = try StreamBuffer.init(allocator, 10000),
            .knowledge_index = KnowledgeGraphIndex.init(allocator),
            .validation_threshold = 0.5,
            .extraction_count = 0,
            .validation_count = 0,
            .integration_count = 0,
            .conflict_count = 0,
            .allocator = allocator,
            .start_time = @as(i64, @truncate(std.time.nanoTimestamp())),
            .total_confidence_sum = 0.0,
            .relation_patterns = ArrayList(RelationPattern).init(allocator),
            .tokenizer_config = TokenizerConfig.default(),
            .relation_statistics = StringHashMap(RelationStatistics).init(allocator),
            .entity_statistics = StringHashMap(EntityStatistics).init(allocator),
            .is_running = true,
            .inference_hooks = ArrayList(InferenceHook).init(allocator),
        };
        errdefer pipeline.deinit();
        try pipeline.initializeDefaultPatterns();
        return pipeline;
    }

    fn initializeDefaultPatterns(self: *CREVPipeline) !void {
        try self.relation_patterns.append(.{ .pattern = " is a ", .relation_type = "is_a", .weight = 0.9 });
        try self.relation_patterns.append(.{ .pattern = " is ", .relation_type = "is", .weight = 0.7 });
        try self.relation_patterns.append(.{ .pattern = " has ", .relation_type = "has", .weight = 0.8 });
        try self.relation_patterns.append(.{ .pattern = " contains ", .relation_type = "contains", .weight = 0.85 });
        try self.relation_patterns.append(.{ .pattern = " belongs to ", .relation_type = "belongs_to", .weight = 0.85 });
        try self.relation_patterns.append(.{ .pattern = " part of ", .relation_type = "part_of", .weight = 0.85 });
        try self.relation_patterns.append(.{ .pattern = " located in ", .relation_type = "located_in", .weight = 0.8 });
        try self.relation_patterns.append(.{ .pattern = " works at ", .relation_type = "works_at", .weight = 0.8 });
        try self.relation_patterns.append(.{ .pattern = " created ", .relation_type = "created", .weight = 0.75 });
        try self.relation_patterns.append(.{ .pattern = " owns ", .relation_type = "owns", .weight = 0.8 });
        try self.relation_patterns.append(.{ .pattern = " uses ", .relation_type = "uses", .weight = 0.7 });
        try self.relation_patterns.append(.{ .pattern = " produces ", .relation_type = "produces", .weight = 0.75 });
        try self.relation_patterns.append(.{ .pattern = " causes ", .relation_type = "causes", .weight = 0.7 });
        try self.relation_patterns.append(.{ .pattern = " leads to ", .relation_type = "leads_to", .weight = 0.7 });
        try self.relation_patterns.append(.{ .pattern = " related to ", .relation_type = "related_to", .weight = 0.5 });
    }

    pub fn deinit(self: *CREVPipeline) void {
        self.is_running = false;
        self.triplet_buffer.deinit();
        self.knowledge_index.deinit();
        self.relation_patterns.deinit();

        var rel_it = self.relation_statistics.iterator();
        while (rel_it.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
        }
        self.relation_statistics.deinit();

        var ent_it = self.entity_statistics.iterator();
        while (ent_it.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
        }
        self.entity_statistics.deinit();

        self.inference_hooks.deinit();
    }

    pub fn processTextStream(self: *CREVPipeline, text: []const u8) !PipelineResult {
        const start_ns = @as(i64, @truncate(std.time.nanoTimestamp()));
        var result = PipelineResult.init();

        var triplets = try self.extractTriplets(text);
        defer triplets.deinit();

        result.triplets_extracted = triplets.items.len;
        self.extraction_count += triplets.items.len;

        for (triplets.items) |triplet| {
            var validation_result = try self.validateTriplet(triplet);
            defer validation_result.deinit();

            self.validation_count += 1;

            if (validation_result.is_valid) {
                result.triplets_validated += 1;
                triplet.confidence = clamp01(validation_result.confidence_adjusted);

                var integrated_triplet: *RelationalTriplet = triplet;

                if (validation_result.hasConflicts()) {
                    const resolved = try self.resolveConflicts(triplet, validation_result.conflicts.items);
                    result.conflicts_resolved += validation_result.conflictCount();
                    self.conflict_count += validation_result.conflictCount();

                    if (resolved != triplet) {
                        triplet.deinit();
                        self.allocator.destroy(triplet);
                        integrated_triplet = resolved;
                    }
                }

                try self.integrateTriplet(integrated_triplet);
                result.triplets_integrated += 1;
                self.integration_count += 1;
            } else {
                triplet.deinit();
                self.allocator.destroy(triplet);
            }
        }

        const end_ns = @as(i64, @truncate(std.time.nanoTimestamp()));
        result.processing_time_ns = end_ns - start_ns;
        result.stage = .indexing;
        return result;
    }

    pub fn processStructuredDataStream(self: *CREVPipeline, data: []const u8) !PipelineResult {
        const start_ns = @as(i64, @truncate(std.time.nanoTimestamp()));
        var result = PipelineResult.init();

        var triplets = try self.extractTripletsFromStructured(data);
        defer triplets.deinit();

        result.triplets_extracted = triplets.items.len;
        self.extraction_count += triplets.items.len;

        for (triplets.items) |triplet| {
            var validation_result = try self.validateTriplet(triplet);
            defer validation_result.deinit();

            self.validation_count += 1;

            if (validation_result.is_valid) {
                result.triplets_validated += 1;
                triplet.confidence = clamp01(validation_result.confidence_adjusted);

                try self.integrateTriplet(triplet);
                result.triplets_integrated += 1;
                self.integration_count += 1;
            } else {
                triplet.deinit();
                self.allocator.destroy(triplet);
            }
        }

        const end_ns = @as(i64, @truncate(std.time.nanoTimestamp()));
        result.processing_time_ns = end_ns - start_ns;
        result.stage = .indexing;
        return result;
    }

    pub fn processImageMetadataStream(self: *CREVPipeline, metadata: []const u8) !PipelineResult {
        const start_ns = @as(i64, @truncate(std.time.nanoTimestamp()));
        var result = PipelineResult.init();

        var triplets = try self.extractTripletsFromImageMetadata(metadata);
        defer triplets.deinit();

        result.triplets_extracted = triplets.items.len;
        self.extraction_count += triplets.items.len;

        for (triplets.items) |triplet| {
            var validation_result = try self.validateTriplet(triplet);
            defer validation_result.deinit();

            self.validation_count += 1;

            if (validation_result.is_valid) {
                result.triplets_validated += 1;
                triplet.confidence = clamp01(validation_result.confidence_adjusted);

                try self.integrateTriplet(triplet);
                result.triplets_integrated += 1;
                self.integration_count += 1;
            } else {
                triplet.deinit();
                self.allocator.destroy(triplet);
            }
        }

        const end_ns = @as(i64, @truncate(std.time.nanoTimestamp()));
        result.processing_time_ns = end_ns - start_ns;
        result.stage = .indexing;
        return result;
    }

    pub fn extractTriplets(self: *CREVPipeline, text: []const u8) !ArrayList(*RelationalTriplet) {
        var triplets = ArrayList(*RelationalTriplet).init(self.allocator);
        errdefer {
            for (triplets.items) |t| {
                t.deinit();
                self.allocator.destroy(t);
            }
            triplets.deinit();
        }

        var sentences = ArrayList([]const u8).init(self.allocator);
        defer sentences.deinit();

        var start: usize = 0;
        var i: usize = 0;
        while (i < text.len) : (i += 1) {
            const c = text[i];
            if (c == '.' or c == '!' or c == '?' or c == '\n') {
                if (i > start) {
                    const sentence = std.mem.trim(u8, text[start..i], " \t\r\n");
                    if (sentence.len >= self.tokenizer_config.min_entity_length) {
                        try sentences.append(sentence);
                    }
                }
                start = i + 1;
            }
        }
        if (start < text.len) {
            const sentence = std.mem.trim(u8, text[start..], " \t\r\n");
            if (sentence.len >= self.tokenizer_config.min_entity_length) {
                try sentences.append(sentence);
            }
        }

        for (sentences.items) |sentence| {
            var best_match: ?struct { match_start: usize, match_end: usize, pat: RelationPattern } = null;
            for (self.relation_patterns.items) |pattern| {
                if (matchPatternMorphemeAware(sentence, pattern.pattern, self.allocator)) |m| {
                    const match_len = m.match_end - m.match_start;
                    const best_len = if (best_match) |bm| bm.match_end - bm.match_start else @as(usize, 0);
                    if (best_match == null or match_len > best_len) {
                        best_match = .{ .match_start = m.match_start, .match_end = m.match_end, .pat = pattern };
                    }
                }
            }

            if (best_match) |m| {
                const subject = std.mem.trim(u8, sentence[0..m.match_start], " \t\r\n,;:");
                if (m.match_end < sentence.len) {
                    const object = std.mem.trim(u8, sentence[m.match_end..], " \t\r\n.,;:!?");
                    const pattern = m.pat;

                    if (subject.len >= self.tokenizer_config.min_entity_length and
                        subject.len <= self.tokenizer_config.max_entity_length and
                        object.len >= self.tokenizer_config.min_entity_length and
                        object.len <= self.tokenizer_config.max_entity_length and
                        pattern.relation_type.len > 0)
                    {
                        const confidence = clamp01(pattern.weight) * self.computeConfidence(subject, object);
                        if (confidence >= self.tokenizer_config.min_confidence_threshold) {
                            const triplet = try self.allocator.create(RelationalTriplet);
                            errdefer self.allocator.destroy(triplet);
                            triplet.* = try RelationalTriplet.init(
                                self.allocator,
                                subject,
                                pattern.relation_type,
                                object,
                                confidence,
                            );
                            try triplets.append(triplet);
                        }
                    }
                }
            }
        }

        return triplets;
    }

    fn extractTripletsFromStructured(self: *CREVPipeline, data: []const u8) !ArrayList(*RelationalTriplet) {
        var triplets = ArrayList(*RelationalTriplet).init(self.allocator);
        errdefer {
            for (triplets.items) |t| {
                t.deinit();
                self.allocator.destroy(t);
            }
            triplets.deinit();
        }

        var lines = std.mem.splitSequence(u8, data, "\n");
        while (lines.next()) |line| {
            const trimmed = std.mem.trim(u8, line, " \t\r");
            if (trimmed.len == 0) continue;

            var parts = std.mem.splitSequence(u8, trimmed, ",");
            var fields = ArrayList([]const u8).init(self.allocator);
            defer fields.deinit();

            while (parts.next()) |part| {
                try fields.append(std.mem.trim(u8, part, " \t\""));
            }

            if (fields.items.len >= 3) {
                const conf = if (fields.items.len >= 4)
                    clamp01(std.fmt.parseFloat(f64, fields.items[3]) catch 0.8)
                else
                    0.8;

                const subj = fields.items[0];
                const rel = fields.items[1];
                const obj = fields.items[2];

                if (subj.len == 0 or rel.len == 0 or obj.len == 0) continue;

                const triplet = try self.allocator.create(RelationalTriplet);
                errdefer self.allocator.destroy(triplet);
                triplet.* = try RelationalTriplet.init(
                    self.allocator,
                    subj,
                    rel,
                    obj,
                    conf,
                );
                try triplets.append(triplet);
            }
        }

        return triplets;
    }

    fn extractTripletsFromImageMetadata(self: *CREVPipeline, metadata: []const u8) !ArrayList(*RelationalTriplet) {
        var triplets = ArrayList(*RelationalTriplet).init(self.allocator);
        errdefer {
            for (triplets.items) |t| {
                t.deinit();
                self.allocator.destroy(t);
            }
            triplets.deinit();
        }

        var lines = std.mem.splitSequence(u8, metadata, "\n");
        while (lines.next()) |line| {
            const trimmed = std.mem.trim(u8, line, " \t\r");
            if (trimmed.len == 0) continue;

            if (std.mem.indexOfScalar(u8, trimmed, ':')) |colon_pos| {
                const key = std.mem.trim(u8, trimmed[0..colon_pos], " \t");
                const value = std.mem.trim(u8, trimmed[colon_pos + 1 ..], " \t");
                if (key.len == 0 or value.len == 0) continue;

                const triplet = try self.allocator.create(RelationalTriplet);
                errdefer self.allocator.destroy(triplet);
                triplet.* = try RelationalTriplet.init(
                    self.allocator,
                    "image",
                    key,
                    value,
                    0.9,
                );
                try triplet.setMetadata("source_type", "image_metadata");
                try triplets.append(triplet);
            }
        }

        return triplets;
    }

    fn computeConfidence(self: *CREVPipeline, subject: []const u8, object: []const u8) f64 {
        _ = self;
        var confidence: f64 = 1.0;

        const subject_len = @as(f64, @floatFromInt(subject.len));
        const object_len = @as(f64, @floatFromInt(object.len));

        if (subject_len < 3.0 or object_len < 3.0) {
            confidence *= 0.7;
        }

        if (subject_len > 50.0 or object_len > 50.0) {
            confidence *= 0.85;
        }

        var subject_upper: usize = 0;
        for (subject) |c| {
            if (c >= 'A' and c <= 'Z') subject_upper += 1;
        }
        if (subject.len > 0 and subject_upper == subject.len) {
            confidence *= 0.9;
        } else if (subject.len > 0 and subject[0] >= 'A' and subject[0] <= 'Z') {
            confidence *= 1.05;
        }

        return clamp01(confidence);
    }

    pub fn validateTriplet(self: *CREVPipeline, triplet: *RelationalTriplet) !ValidationResult {
        var result = ValidationResult.init(self.allocator, triplet);

        if (triplet.subject.len < self.tokenizer_config.min_entity_length or
            triplet.object.len < self.tokenizer_config.min_entity_length or
            triplet.relation.len == 0)
        {
            result.is_valid = false;
            result.setValidationMethod("basic_checks");
            return result;
        }

        triplet.confidence = clamp01(triplet.confidence);

        if (triplet.confidence < self.validation_threshold) {
            result.is_valid = false;
            result.confidence_adjusted = triplet.confidence;
            result.setValidationMethod("confidence_threshold");
            return result;
        }

        var existing_triplets = try self.knowledge_index.query(triplet.subject, null, triplet.object, self.allocator);
        defer existing_triplets.deinit();

        for (existing_triplets.items) |existing| {
            if (!self.checkConsistency(triplet, existing)) {
                try result.addConflict(existing);
            }
        }

        result.anomaly_score = try self.computeAnomalyScore(triplet);

        if (result.anomaly_score > 0.85) {
            result.is_valid = false;
            result.setValidationMethod("anomaly_detection");
            return result;
        }

        var adjusted = triplet.confidence * (1.0 - result.anomaly_score * 0.3);
        if (result.hasConflicts()) adjusted *= 0.9;
        result.confidence_adjusted = clamp01(adjusted);

        result.setValidationMethod("full_validation");
        return result;
    }

    fn computeAnomalyScore(self: *CREVPipeline, triplet: *RelationalTriplet) !f64 {
        var weighted_sum: f64 = 0.0;
        var total_weight: f64 = 0.0;

        if (self.relation_statistics.get(triplet.relation)) |stats| {
            if (stats.count > 10) {
                const std_dev = stats.getStdDev();
                if (std_dev > 0.0) {
                    const z = @abs(triplet.confidence - stats.avg_confidence) / std_dev;
                    const a = @min(1.0, z / 3.0);
                    const w = 0.3;
                    weighted_sum += a * w;
                    total_weight += w;
                }
            }
        }

        const subject_known = self.entity_statistics.contains(triplet.subject);
        const object_known = self.entity_statistics.contains(triplet.object);

        if (self.entity_statistics.count() > 0) {
            if (!subject_known and !object_known) {
                const w = 0.4;
                weighted_sum += 1.0 * w;
                total_weight += w;
            } else if (!subject_known or !object_known) {
                const w = 0.2;
                weighted_sum += 1.0 * w;
                total_weight += w;
            }
        }

        if (self.relation_statistics.count() > 0 and !self.relation_statistics.contains(triplet.relation)) {
            const w = 0.15;
            weighted_sum += 1.0 * w;
            total_weight += w;
        }

        if (total_weight == 0.0) return 0.0;
        return clamp01(weighted_sum / total_weight);
    }

    pub fn checkConsistency(self: *CREVPipeline, triplet: *RelationalTriplet, existing: *RelationalTriplet) bool {
        _ = self;

        if (std.mem.eql(u8, triplet.relation, existing.relation)) {
            return true;
        }

        const contradicting_pairs = [_][2][]const u8{
            .{ "is_a", "is_not" },
            .{ "has", "lacks" },
            .{ "owns", "does_not_own" },
            .{ "contains", "excludes" },
            .{ "causes", "prevents" },
        };

        for (contradicting_pairs) |pair| {
            if ((std.mem.eql(u8, triplet.relation, pair[0]) and std.mem.eql(u8, existing.relation, pair[1])) or
                (std.mem.eql(u8, triplet.relation, pair[1]) and std.mem.eql(u8, existing.relation, pair[0])))
            {
                return false;
            }
        }

        return true;
    }

    pub fn resolveConflicts(
        self: *CREVPipeline,
        triplet: *RelationalTriplet,
        conflicts: []*RelationalTriplet,
    ) !*RelationalTriplet {
        if (conflicts.len == 0) return triplet;

        var best = triplet;
        for (conflicts) |c| {
            if (c.confidence > best.confidence) best = c;
        }

        if (best == triplet) return triplet;

        const new_triplet = try self.allocator.create(RelationalTriplet);
        errdefer self.allocator.destroy(new_triplet);
        new_triplet.* = try best.clone(self.allocator);

        const a = clamp01(triplet.confidence);
        const b = clamp01(best.confidence);
        const denom = a + b;
        new_triplet.confidence = if (denom > 0.0) (a * a + b * b) / denom else b;
        new_triplet.confidence = clamp01(new_triplet.confidence);

        return new_triplet;
    }

    pub fn integrateTriplet(self: *CREVPipeline, triplet: *RelationalTriplet) !void {
        try self.knowledge_index.index(triplet);
        self.total_confidence_sum += triplet.confidence;

        try self.updateStatistics(triplet);

        if (!self.triplet_buffer.push(triplet)) {
            _ = self.triplet_buffer.pop();
            _ = self.triplet_buffer.push(triplet);
        }

        const data = try std.fmt.allocPrint(self.allocator, "{s}|{s}|{s}|{d:.6}", .{
            triplet.subject,
            triplet.relation,
            triplet.object,
            triplet.confidence,
        });
        defer self.allocator.free(data);

        _ = try self.kernel.allocateMemory(data, null);
    }

    fn updateStatistics(self: *CREVPipeline, triplet: *RelationalTriplet) !void {
        const rel_gop = try self.relation_statistics.getOrPut(triplet.relation);
        if (!rel_gop.found_existing) {
            const key_copy = try self.allocator.dupe(u8, triplet.relation);
            errdefer self.allocator.free(key_copy);
            rel_gop.key_ptr.* = key_copy;
            rel_gop.value_ptr.* = RelationStatistics.init();
        }
        rel_gop.value_ptr.*.update(triplet.confidence);

        const subj_gop = try self.entity_statistics.getOrPut(triplet.subject);
        if (!subj_gop.found_existing) {
            const key_copy = try self.allocator.dupe(u8, triplet.subject);
            errdefer self.allocator.free(key_copy);
            subj_gop.key_ptr.* = key_copy;
            subj_gop.value_ptr.* = EntityStatistics.init();
        }
        subj_gop.value_ptr.*.count += 1;
        subj_gop.value_ptr.*.as_subject += 1;
        subj_gop.value_ptr.*.total_confidence += triplet.confidence;

        const obj_gop = try self.entity_statistics.getOrPut(triplet.object);
        if (!obj_gop.found_existing) {
            const key_copy = try self.allocator.dupe(u8, triplet.object);
            errdefer self.allocator.free(key_copy);
            obj_gop.key_ptr.* = key_copy;
            obj_gop.value_ptr.* = EntityStatistics.init();
        }
        obj_gop.value_ptr.*.count += 1;
        obj_gop.value_ptr.*.as_object += 1;
        obj_gop.value_ptr.*.total_confidence += triplet.confidence;
    }

    pub fn queryKnowledgeGraph(
        self: *CREVPipeline,
        subject: ?[]const u8,
        relation: ?[]const u8,
        object: ?[]const u8,
    ) !ArrayList(*RelationalTriplet) {
        return self.knowledge_index.query(subject, relation, object, self.allocator);
    }

    pub fn getPipelineStatistics(self: *CREVPipeline) PipelineStatistics {
        const uptime_ns = @as(i64, @truncate(std.time.nanoTimestamp())) - self.start_time;
        const uptime_ms_i64: i64 = @as(i64, @intCast(@max(@as(i128, 0), @divTrunc(uptime_ns, 1_000_000))));
        const uptime_sec = @as(f64, @floatFromInt(@max(@as(i128, 1), uptime_ns))) / 1_000_000_000.0;

        return PipelineStatistics{
            .total_extractions = self.extraction_count,
            .total_validations = self.validation_count,
            .total_integrations = self.integration_count,
            .average_confidence = if (self.integration_count > 0)
                self.total_confidence_sum / @as(f64, @floatFromInt(self.integration_count))
            else
                0.0,
            .conflict_rate = if (self.validation_count > 0)
                @as(f64, @floatFromInt(self.conflict_count)) / @as(f64, @floatFromInt(self.validation_count))
            else
                0.0,
            .throughput = @as(f64, @floatFromInt(self.integration_count)) / uptime_sec,
            .buffer_utilization = self.triplet_buffer.getUtilization(),
            .unique_subjects = self.knowledge_index.getUniqueSubjects(),
            .unique_relations = self.knowledge_index.getUniqueRelations(),
            .unique_objects = self.knowledge_index.getUniqueObjects(),
            .uptime_ms = uptime_ms_i64,
        };
    }

    pub fn shutdown(self: *CREVPipeline) void {
        self.is_running = false;
        self.triplet_buffer.clear();
    }

    pub fn addRelationPattern(self: *CREVPipeline, pattern: []const u8, relation_type: []const u8, weight_in: f64) !void {
        const p_copy = try self.allocator.dupe(u8, pattern);
        errdefer self.allocator.free(p_copy);
        const r_copy = try self.allocator.dupe(u8, relation_type);
        errdefer self.allocator.free(r_copy);

        try self.relation_patterns.append(.{
            .pattern = p_copy,
            .relation_type = r_copy,
            .weight = clamp01(weight_in),
        });
    }

    pub fn setValidationThreshold(self: *CREVPipeline, threshold: f64) void {
        self.validation_threshold = clamp01(threshold);
    }

    pub fn getKnowledgeGraphSize(self: *CREVPipeline) usize {
        return self.knowledge_index.count();
    }

    pub fn registerInferenceHook(self: *CREVPipeline, hook: InferenceHook) !void {
        try self.inference_hooks.append(hook);
    }

    pub fn clearInferenceHooks(self: *CREVPipeline) void {
        self.inference_hooks.clearRetainingCapacity();
    }

    pub fn processInferenceText(self: *CREVPipeline, text: []const u8) !PipelineResult {
        for (self.inference_hooks.items) |hook| {
            if (hook.pre_process) |cb| {
                cb(hook.context, text);
            }
        }

        var result = try self.processTextStream(text);

        for (self.inference_hooks.items) |hook| {
            if (hook.post_process) |cb| {
                cb(hook.context, &result);
            }
        }

        return result;
    }

    pub fn queryInferenceKnowledge(self: *CREVPipeline, subject: ?[]const u8, relation: ?[]const u8, object: ?[]const u8) !ArrayList(*RelationalTriplet) {
        for (self.inference_hooks.items) |hook| {
            if (hook.pre_query) |cb| {
                cb(hook.context, subject, relation, object);
            }
        }

        const results = try self.knowledge_index.queryMorphemeAware(subject, relation, object, self.allocator);

        for (self.inference_hooks.items) |hook| {
            if (hook.post_query) |cb| {
                cb(hook.context, results.items.len);
            }
        }

        return results;
    }

    pub fn getInferencePipelineStatistics(self: *CREVPipeline) PipelineStatistics {
        return self.getPipelineStatistics();
    }

    pub fn isRunning(self: *const CREVPipeline) bool {
        return self.is_running;
    }
};

test "ExtractionStage toString and fromString" {
    const testing = std.testing;

    try testing.expectEqualStrings("tokenization", ExtractionStage.tokenization.toString());
    try testing.expectEqualStrings("triplet_extraction", ExtractionStage.triplet_extraction.toString());
    try testing.expectEqualStrings("validation", ExtractionStage.validation.toString());
    try testing.expectEqualStrings("integration", ExtractionStage.integration.toString());
    try testing.expectEqualStrings("indexing", ExtractionStage.indexing.toString());

    try testing.expectEqual(ExtractionStage.tokenization, ExtractionStage.fromString("tokenization").?);
    try testing.expectEqual(ExtractionStage.validation, ExtractionStage.fromString("validation").?);
    try testing.expect(ExtractionStage.fromString("invalid") == null);
}

test "ExtractionStage next" {
    const testing = std.testing;

    try testing.expectEqual(ExtractionStage.triplet_extraction, ExtractionStage.tokenization.next().?);
    try testing.expectEqual(ExtractionStage.validation, ExtractionStage.triplet_extraction.next().?);
    try testing.expectEqual(ExtractionStage.integration, ExtractionStage.validation.next().?);
    try testing.expectEqual(ExtractionStage.indexing, ExtractionStage.integration.next().?);
    try testing.expect(ExtractionStage.indexing.next() == null);
}

test "RelationalTriplet initialization" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var triplet = try RelationalTriplet.init(allocator, "Alice", "knows", "Bob", 0.9);
    defer triplet.deinit();

    try testing.expectEqualStrings("Alice", triplet.subject);
    try testing.expectEqualStrings("knows", triplet.relation);
    try testing.expectEqualStrings("Bob", triplet.object);
    try testing.expectApproxEqAbs(@as(f64, 0.9), triplet.confidence, 0.001);
}

test "RelationalTriplet clone" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var original = try RelationalTriplet.init(allocator, "Paris", "is_a", "City", 0.95);
    defer original.deinit();

    try original.setMetadata("source", "test");

    var cloned = try original.clone(allocator);
    defer cloned.deinit();

    try testing.expectEqualStrings(original.subject, cloned.subject);
    try testing.expectEqualStrings(original.relation, cloned.relation);
    try testing.expectEqualStrings(original.object, cloned.object);
    try testing.expectApproxEqAbs(original.confidence, cloned.confidence, 0.001);
    try testing.expectEqualStrings("test", cloned.getMetadata("source").?);
}

test "RelationalTriplet computeHash" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var triplet1 = try RelationalTriplet.init(allocator, "A", "B", "C", 0.5);
    defer triplet1.deinit();

    var triplet2 = try RelationalTriplet.init(allocator, "A", "B", "C", 0.5);
    defer triplet2.deinit();

    const hash1 = triplet1.computeHash();
    const hash2 = triplet2.computeHash();

    try testing.expect(hash1.len == 32);
    try testing.expect(hash2.len == 32);
}

test "RelationalTriplet equals" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var triplet1 = try RelationalTriplet.init(allocator, "A", "rel", "B", 0.9);
    defer triplet1.deinit();

    var triplet2 = try RelationalTriplet.init(allocator, "A", "rel", "B", 0.8);
    defer triplet2.deinit();

    var triplet3 = try RelationalTriplet.init(allocator, "A", "different", "B", 0.9);
    defer triplet3.deinit();

    try testing.expect(triplet1.equals(&triplet2));
    try testing.expect(!triplet1.equals(&triplet3));
}

test "KnowledgeGraphIndex initialization and indexing" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var index = KnowledgeGraphIndex.init(allocator);
    defer index.deinit();

    const triplet = try allocator.create(RelationalTriplet);
    triplet.* = try RelationalTriplet.init(allocator, "Entity1", "related_to", "Entity2", 0.8);

    try index.index(triplet);

    try testing.expectEqual(@as(usize, 1), index.count());
}

test "KnowledgeGraphIndex query" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var index = KnowledgeGraphIndex.init(allocator);
    defer index.deinit();

    const triplet1 = try allocator.create(RelationalTriplet);
    triplet1.* = try RelationalTriplet.init(allocator, "Alice", "knows", "Bob", 0.9);
    try index.index(triplet1);

    const triplet2 = try allocator.create(RelationalTriplet);
    triplet2.* = try RelationalTriplet.init(allocator, "Alice", "works_at", "Company", 0.85);
    try index.index(triplet2);

    var results = try index.query("Alice", null, null, allocator);
    defer results.deinit();

    try testing.expectEqual(@as(usize, 2), results.items.len);
}

test "KnowledgeGraphIndex queryBySubject" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var index = KnowledgeGraphIndex.init(allocator);
    defer index.deinit();

    const triplet = try allocator.create(RelationalTriplet);
    triplet.* = try RelationalTriplet.init(allocator, "TestSubject", "has", "TestObject", 0.7);
    try index.index(triplet);

    const results = index.queryBySubject("TestSubject");
    try testing.expectEqual(@as(usize, 1), results.len);

    const empty_results = index.queryBySubject("NonExistent");
    try testing.expectEqual(@as(usize, 0), empty_results.len);
}

test "KnowledgeGraphIndex remove" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var index = KnowledgeGraphIndex.init(allocator);
    defer index.deinit();

    const triplet = try allocator.create(RelationalTriplet);
    triplet.* = try RelationalTriplet.init(allocator, "ToRemove", "relation", "Target", 0.6);
    try index.index(triplet);

    try testing.expectEqual(@as(usize, 1), index.count());

    const removed = index.remove(triplet);
    try testing.expect(removed);
    try testing.expectEqual(@as(usize, 0), index.count());

    triplet.deinit();
    allocator.destroy(triplet);
}

test "StreamBuffer push and pop" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var buffer = try StreamBuffer.init(allocator, 5);
    defer buffer.deinit();

    try testing.expect(buffer.isEmpty());
    try testing.expect(!buffer.isFull());

    const triplet1 = try allocator.create(RelationalTriplet);
    triplet1.* = try RelationalTriplet.init(allocator, "A", "B", "C", 0.5);
    const ok = buffer.push(triplet1);
    try testing.expect(ok);

    try testing.expect(!buffer.isEmpty());
    try testing.expectEqual(@as(usize, 1), buffer.getSize());

    const popped = buffer.pop();
    try testing.expect(popped != null);
    try testing.expectEqualStrings("A", popped.?.subject);
    try testing.expect(buffer.isEmpty());

    popped.?.deinit();
    allocator.destroy(popped.?);
}

test "StreamBuffer capacity" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var buffer = try StreamBuffer.init(allocator, 3);
    defer buffer.deinit();

    const triplet1 = try allocator.create(RelationalTriplet);
    triplet1.* = try RelationalTriplet.init(allocator, "1", "r", "a", 0.5);
    try testing.expect(buffer.push(triplet1));

    const triplet2 = try allocator.create(RelationalTriplet);
    triplet2.* = try RelationalTriplet.init(allocator, "2", "r", "b", 0.5);
    try testing.expect(buffer.push(triplet2));

    const triplet3 = try allocator.create(RelationalTriplet);
    triplet3.* = try RelationalTriplet.init(allocator, "3", "r", "c", 0.5);
    try testing.expect(buffer.push(triplet3));

    try testing.expect(buffer.isFull());

    const triplet4 = try allocator.create(RelationalTriplet);
    triplet4.* = try RelationalTriplet.init(allocator, "4", "r", "d", 0.5);
    const success = buffer.push(triplet4);
    try testing.expect(!success);
    try testing.expectEqual(@as(usize, 1), buffer.overflow_count);

    triplet4.deinit();
    allocator.destroy(triplet4);

    while (buffer.pop()) |t| {
        t.deinit();
        allocator.destroy(t);
    }
}

test "StreamBuffer peek" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var buffer = try StreamBuffer.init(allocator, 5);
    defer buffer.deinit();

    try testing.expect(buffer.peek() == null);

    const triplet = try allocator.create(RelationalTriplet);
    triplet.* = try RelationalTriplet.init(allocator, "Peek", "test", "value", 0.7);
    try testing.expect(buffer.push(triplet));

    const peeked = buffer.peek();
    try testing.expect(peeked != null);
    try testing.expectEqualStrings("Peek", peeked.?.subject);
    try testing.expectEqual(@as(usize, 1), buffer.getSize());

    const popped = buffer.pop().?;
    popped.deinit();
    allocator.destroy(popped);
}

test "PipelineResult merge" {
    const testing = std.testing;

    var result1 = PipelineResult.init();
    result1.triplets_extracted = 10;
    result1.triplets_validated = 8;
    result1.triplets_integrated = 7;

    var result2 = PipelineResult.init();
    result2.triplets_extracted = 5;
    result2.triplets_validated = 4;
    result2.triplets_integrated = 3;

    result1.merge(result2);

    try testing.expectEqual(@as(usize, 15), result1.triplets_extracted);
    try testing.expectEqual(@as(usize, 12), result1.triplets_validated);
    try testing.expectEqual(@as(usize, 10), result1.triplets_integrated);
}

test "CREVPipeline initialization" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var kernel = ChaosCoreKernel.init(allocator);
    defer kernel.deinit();

    var pipeline = try CREVPipeline.init(allocator, &kernel);
    defer pipeline.deinit();

    try testing.expect(pipeline.is_running);
    try testing.expectEqual(@as(usize, 0), pipeline.extraction_count);
    try testing.expectApproxEqAbs(@as(f64, 0.5), pipeline.validation_threshold, 0.001);
}

test "CREVPipeline extractTriplets" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var kernel = ChaosCoreKernel.init(allocator);
    defer kernel.deinit();

    var pipeline = try CREVPipeline.init(allocator, &kernel);
    defer pipeline.deinit();

    const text = "Paris is a city. The Eiffel Tower is located in Paris.";
    var triplets = try pipeline.extractTriplets(text);
    defer {
        for (triplets.items) |triplet| {
            triplet.deinit();
            allocator.destroy(triplet);
        }
        triplets.deinit();
    }

    try testing.expect(triplets.items.len > 0);
}

test "CREVPipeline processTextStream" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var kernel = ChaosCoreKernel.init(allocator);
    defer kernel.deinit();

    var pipeline = try CREVPipeline.init(allocator, &kernel);
    defer pipeline.deinit();

    const text = "Python is a programming language. Python has modules.";
    const result = try pipeline.processTextStream(text);

    try testing.expect(result.triplets_extracted > 0);
    try testing.expect(result.processing_time_ns >= 0);
}

test "CREVPipeline processStructuredDataStream" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var kernel = ChaosCoreKernel.init(allocator);
    defer kernel.deinit();

    var pipeline = try CREVPipeline.init(allocator, &kernel);
    defer pipeline.deinit();

    const data = "Alice,knows,Bob,0.9\nBob,works_at,Company,0.85";
    const result = try pipeline.processStructuredDataStream(data);

    try testing.expect(result.triplets_extracted == 2);
}

test "CREVPipeline validateTriplet" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var kernel = ChaosCoreKernel.init(allocator);
    defer kernel.deinit();

    var pipeline = try CREVPipeline.init(allocator, &kernel);
    defer pipeline.deinit();

    pipeline.setValidationThreshold(0.4);

    const triplet = try allocator.create(RelationalTriplet);
    triplet.* = try RelationalTriplet.init(allocator, "TestSubjectEntity", "is_a", "TestObjectEntity", 0.95);
    defer {
        triplet.deinit();
        allocator.destroy(triplet);
    }

    var result = try pipeline.validateTriplet(triplet);
    defer result.deinit();

    try testing.expect(result.confidence_adjusted > 0);
}

test "CREVPipeline checkConsistency" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var kernel = ChaosCoreKernel.init(allocator);
    defer kernel.deinit();

    var pipeline = try CREVPipeline.init(allocator, &kernel);
    defer pipeline.deinit();

    var triplet1 = try RelationalTriplet.init(allocator, "A", "is_a", "B", 0.9);
    defer triplet1.deinit();

    var triplet2 = try RelationalTriplet.init(allocator, "A", "is_a", "B", 0.8);
    defer triplet2.deinit();

    try testing.expect(pipeline.checkConsistency(&triplet1, &triplet2));

    var triplet3 = try RelationalTriplet.init(allocator, "A", "is_not", "B", 0.7);
    defer triplet3.deinit();

    try testing.expect(!pipeline.checkConsistency(&triplet1, &triplet3));
}

test "CREVPipeline getPipelineStatistics" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var kernel = ChaosCoreKernel.init(allocator);
    defer kernel.deinit();

    var pipeline = try CREVPipeline.init(allocator, &kernel);
    defer pipeline.deinit();

    const stats = pipeline.getPipelineStatistics();

    try testing.expectEqual(@as(usize, 0), stats.total_extractions);
    try testing.expectEqual(@as(usize, 0), stats.total_validations);
    try testing.expect(stats.uptime_ms >= 0);
}

test "CREVPipeline queryKnowledgeGraph" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var kernel = ChaosCoreKernel.init(allocator);
    defer kernel.deinit();

    var pipeline = try CREVPipeline.init(allocator, &kernel);
    defer pipeline.deinit();

    const triplet = try allocator.create(RelationalTriplet);
    triplet.* = try RelationalTriplet.init(allocator, "DirectEntity", "has_property", "TestProperty", 0.9);
    try pipeline.knowledge_index.index(triplet);

    var results = try pipeline.queryKnowledgeGraph("DirectEntity", null, null);
    defer results.deinit();

    try testing.expect(results.items.len > 0);
}

test "CREVPipeline shutdown" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var kernel = ChaosCoreKernel.init(allocator);
    defer kernel.deinit();

    var pipeline = try CREVPipeline.init(allocator, &kernel);
    defer pipeline.deinit();

    try testing.expect(pipeline.isRunning());
    pipeline.shutdown();
    try testing.expect(!pipeline.isRunning());
}

test "ValidationResult initialization" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var triplet = try RelationalTriplet.init(allocator, "S", "R", "O", 0.75);
    defer triplet.deinit();

    var result = ValidationResult.init(allocator, &triplet);
    defer result.deinit();

    try testing.expect(result.is_valid);
    try testing.expect(!result.hasConflicts());
    try testing.expectEqual(@as(usize, 0), result.conflictCount());
}

test "RelationStatistics update" {
    const testing = std.testing;

    var stats = CREVPipeline.RelationStatistics.init();

    stats.update(0.8);
    try testing.expectEqual(@as(usize, 1), stats.count);
    try testing.expectApproxEqAbs(@as(f64, 0.8), stats.avg_confidence, 0.001);

    stats.update(0.6);
    try testing.expectEqual(@as(usize, 2), stats.count);
    try testing.expectApproxEqAbs(@as(f64, 0.7), stats.avg_confidence, 0.001);

    try testing.expect(stats.getVariance() >= 0);
    try testing.expect(stats.getStdDev() >= 0);
}

test "StreamBuffer utilization" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var buffer = try StreamBuffer.init(allocator, 10);
    defer buffer.deinit();

    try testing.expectApproxEqAbs(@as(f64, 0.0), buffer.getUtilization(), 0.001);

    const triplet1 = try allocator.create(RelationalTriplet);
    triplet1.* = try RelationalTriplet.init(allocator, "1", "r", "a", 0.5);
    try testing.expect(buffer.push(triplet1));

    const triplet2 = try allocator.create(RelationalTriplet);
    triplet2.* = try RelationalTriplet.init(allocator, "2", "r", "b", 0.5);
    try testing.expect(buffer.push(triplet2));

    try testing.expectApproxEqAbs(@as(f64, 0.2), buffer.getUtilization(), 0.001);

    while (buffer.pop()) |t| {
        t.deinit();
        allocator.destroy(t);
    }
}


const COG_K: usize = 4;
const COG_H: usize = 16;
const ROUTE_TOP_K: usize = 64;
const SKILL_NUMERIC_SEED: u64 = 0x5EEDF00D5EEDF00D;

const SQLITE_OK: c_int = 0;
const SQLITE_ROW: c_int = 100;
const SQLITE_DONE: c_int = 101;
const SQLITE_OPEN_READWRITE: c_int = 0x00000002;
const SQLITE_OPEN_CREATE: c_int = 0x00000004;
const SQLITE_OPEN_FULLMUTEX: c_int = 0x00010000;
const SQLITE_TRANSIENT: isize = -1;

const sqlite3 = opaque {};
const sqlite3_stmt = opaque {};

extern fn sqlite3_open_v2(filename: [*:0]const u8, ppDb: *?*sqlite3, flags: c_int, zVfs: ?[*:0]const u8) c_int;
extern fn sqlite3_close(db: ?*sqlite3) c_int;
extern fn sqlite3_exec(db: ?*sqlite3, sql: [*:0]const u8, callback: ?*const fn (?*anyopaque, c_int, [*c][*c]u8, [*c][*c]u8) callconv(.C) c_int, arg: ?*anyopaque, errmsg: ?*[*c]u8) c_int;
extern fn sqlite3_prepare_v2(db: ?*sqlite3, zSql: [*]const u8, nByte: c_int, ppStmt: *?*sqlite3_stmt, pzTail: ?*[*c]const u8) c_int;
extern fn sqlite3_step(stmt: ?*sqlite3_stmt) c_int;
extern fn sqlite3_finalize(stmt: ?*sqlite3_stmt) c_int;
extern fn sqlite3_bind_text(stmt: ?*sqlite3_stmt, idx: c_int, val: [*]const u8, n: c_int, destructor: ?*const fn (?*anyopaque) callconv(.C) void) c_int;
extern fn sqlite3_bind_int64(stmt: ?*sqlite3_stmt, idx: c_int, val: i64) c_int;
extern fn sqlite3_bind_int(stmt: ?*sqlite3_stmt, idx: c_int, val: c_int) c_int;
extern fn sqlite3_bind_double(stmt: ?*sqlite3_stmt, idx: c_int, val: f64) c_int;
extern fn sqlite3_bind_null(stmt: ?*sqlite3_stmt, idx: c_int) c_int;
extern fn sqlite3_column_text(stmt: ?*sqlite3_stmt, col: c_int) [*c]const u8;
extern fn sqlite3_column_bytes(stmt: ?*sqlite3_stmt, col: c_int) c_int;
extern fn sqlite3_column_int64(stmt: ?*sqlite3_stmt, col: c_int) i64;
extern fn sqlite3_column_int(stmt: ?*sqlite3_stmt, col: c_int) c_int;
extern fn sqlite3_column_double(stmt: ?*sqlite3_stmt, col: c_int) f64;
extern fn sqlite3_errmsg(db: ?*sqlite3) [*c]const u8;
extern fn sqlite3_last_insert_rowid(db: ?*sqlite3) i64;
extern fn sqlite3_busy_timeout(db: ?*sqlite3, ms: c_int) c_int;
extern fn sqlite3_changes(db: ?*sqlite3) c_int;

const Config = struct {
    host: []u8,
    port: u16,
    database_path: []u8,
    modular_api_key: []u8,
    modular_base_url: []u8,
    model: []u8,
    workspace_root: []u8,
    knowledge_root: []u8,
    default_tenant_id: []u8,
    allowed_http_hosts: []u8,
    max_request_bytes: usize,
    max_state_bytes: usize,
    max_prompt_bytes: usize,
    max_response_bytes: usize,
    max_steps: i64,
    default_token_budget: i64,
    model_max_tokens: i64,

    fn load(allocator: Allocator) !Config {
        const port_text = try envOwnedOr(allocator, "PORT", "8080");
        defer allocator.free(port_text);
        const port = try std.fmt.parseInt(u16, port_text, 10);

        const max_request_text = try envOwnedOr(allocator, "AGENT_MAX_REQUEST_BYTES", "10485760");
        defer allocator.free(max_request_text);
        const max_state_text = try envOwnedOr(allocator, "AGENT_MAX_STATE_BYTES", "65536");
        defer allocator.free(max_state_text);
        const max_prompt_text = try envOwnedOr(allocator, "AGENT_MAX_PROMPT_BYTES", "131072");
        defer allocator.free(max_prompt_text);
        const max_response_text = try envOwnedOr(allocator, "AGENT_MAX_RESPONSE_BYTES", "20971520");
        defer allocator.free(max_response_text);
        const max_steps_text = try envOwnedOr(allocator, "AGENT_MAX_STEPS", "100000");
        defer allocator.free(max_steps_text);
        const token_budget_text = try envOwnedOr(allocator, "AGENT_DEFAULT_TOKEN_BUDGET", "10000000");
        defer allocator.free(token_budget_text);
        const model_max_tokens_text = try envOwnedOr(allocator, "AGENT_MODEL_MAX_TOKENS", "100000");
        defer allocator.free(model_max_tokens_text);

        return Config{
            .host = try envOwnedOr(allocator, "HOST", "0.0.0.0"),
            .port = port,
            .database_path = try envOwnedOr(allocator, "AGENT_DATABASE_PATH", "agent_runtime.sqlite3"),
            .modular_api_key = try std.process.getEnvVarOwned(allocator, "MODULAR_API_KEY"),
            .modular_base_url = try envOwnedOr(allocator, "MODULAR_BASE_URL", "https://api.modular.com/v1"),
            .model = try envOwnedOr(allocator, "MODULAR_MODEL", "zai-org/glm-5.3"),
            .workspace_root = try envOwnedOr(allocator, "AGENT_WORKSPACE", "agent_workspace"),
            .knowledge_root = try envOwnedOr(allocator, "AGENT_KNOWLEDGE_ROOT", "agent_knowledge"),
            .default_tenant_id = try envOwnedOr(allocator, "AGENT_DEFAULT_TENANT", "default"),
            .allowed_http_hosts = try envOwnedOr(allocator, "AGENT_ALLOWED_HTTP_HOSTS", "*"),
            .max_request_bytes = try std.fmt.parseInt(usize, max_request_text, 10),
            .max_state_bytes = try std.fmt.parseInt(usize, max_state_text, 10),
            .max_prompt_bytes = try std.fmt.parseInt(usize, max_prompt_text, 10),
            .max_response_bytes = try std.fmt.parseInt(usize, max_response_text, 10),
            .max_steps = try std.fmt.parseInt(i64, max_steps_text, 10),
            .default_token_budget = try std.fmt.parseInt(i64, token_budget_text, 10),
            .model_max_tokens = try std.fmt.parseInt(i64, model_max_tokens_text, 10),
        };
    }

    fn deinit(self: *Config, allocator: Allocator) void {
        allocator.free(self.host);
        allocator.free(self.database_path);
        allocator.free(self.modular_api_key);
        allocator.free(self.modular_base_url);
        allocator.free(self.model);
        allocator.free(self.workspace_root);
        allocator.free(self.knowledge_root);
        allocator.free(self.default_tenant_id);
        allocator.free(self.allowed_http_hosts);
    }
};

fn envOwnedOr(allocator: Allocator, name: []const u8, default_value: []const u8) ![]u8 {
    return std.process.getEnvVarOwned(allocator, name) catch |err| switch (err) {
        error.EnvironmentVariableNotFound => allocator.dupe(u8, default_value),
        else => err,
    };
}

fn now() i64 {
    return std.time.timestamp();
}

fn nowMillis() i64 {
    return std.time.milliTimestamp();
}

fn makeId(allocator: Allocator, prefix: []const u8) ![]u8 {
    var bytes: [16]u8 = undefined;
    std.crypto.random.bytes(&bytes);
    const hex = "0123456789abcdef";
    var out = try allocator.alloc(u8, prefix.len + 1 + 32);
    @memcpy(out[0..prefix.len], prefix);
    out[prefix.len] = '_';
    for (bytes, 0..) |b, i| {
        out[prefix.len + 1 + i * 2] = hex[(b >> 4) & 0x0f];
        out[prefix.len + 1 + i * 2 + 1] = hex[b & 0x0f];
    }
    return out;
}

const RunRecord = struct {
    id: []u8,
    tenant_id: []u8,
    procedure_json: []u8,
    status: []u8,
    step: i64,
    latest_observation_json: []u8,
    token_budget_used: i64,
    token_budget_limit: i64,

    fn deinit(self: *RunRecord, allocator: Allocator) void {
        allocator.free(self.id);
        allocator.free(self.tenant_id);
        allocator.free(self.procedure_json);
        allocator.free(self.status);
        allocator.free(self.latest_observation_json);
    }
};

const SkillRecord = struct {
    id: []u8,
    name: []u8,
    description: []u8,
    trigger_json: []u8,
    procedure_json: []u8,
    embedding_json: []u8,
    vector_score: f64,
    sparse_rank: usize,
    rrf_score: f64,

    fn deinit(self: *SkillRecord, allocator: Allocator) void {
        allocator.free(self.id);
        allocator.free(self.name);
        allocator.free(self.description);
        allocator.free(self.trigger_json);
        allocator.free(self.procedure_json);
        allocator.free(self.embedding_json);
    }
};

const ActionClaim = struct {
    id: i64,
    run_id: []u8,
    step: i64,
    action_json: []u8,

    fn deinit(self: *ActionClaim, allocator: Allocator) void {
        allocator.free(self.run_id);
        allocator.free(self.action_json);
    }
};

const ModelResult = struct {
    content: []u8,
    total_tokens: i64,

    fn deinit(self: *ModelResult, allocator: Allocator) void {
        allocator.free(self.content);
    }
};

const Envelope = struct {
    envelope_json: []u8,
    patch_json: []u8,
    action_json: []u8,
    terminal: bool,
    confidence: f64,

    fn deinit(self: *Envelope, allocator: Allocator) void {
        allocator.free(self.envelope_json);
        allocator.free(self.patch_json);
        allocator.free(self.action_json);
    }
};

const HttpRequest = struct {
    method: []const u8,
    target: []const u8,
    path: []const u8,
    query: []const u8,
    headers: std.StringHashMap([]const u8),
    body: []const u8,
};

const Database = struct {
    allocator: Allocator,
    handle: ?*sqlite3,
    mutex: std.Thread.Mutex,

    fn open(allocator: Allocator, path: []const u8) !Database {
        const path_z = try allocator.dupeZ(u8, path);
        defer allocator.free(path_z);
        var handle: ?*sqlite3 = null;
        const flags = SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE | SQLITE_OPEN_FULLMUTEX;
        const rc = sqlite3_open_v2(path_z.ptr, &handle, flags, null);
        if (rc != SQLITE_OK) return error.SqliteOpenFailed;
        _ = sqlite3_busy_timeout(handle, 10000);
        return Database{ .allocator = allocator, .handle = handle, .mutex = .{} };
    }

    fn close(self: *Database) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        if (self.handle != null) {
            _ = sqlite3_close(self.handle);
            self.handle = null;
        }
    }

    fn check(self: *Database, rc: c_int) !void {
        _ = self;
        if (rc == SQLITE_OK) return;
        return error.SqliteError;
    }

    fn checkStep(self: *Database, rc: c_int) !void {
        _ = self;
        if (rc == SQLITE_ROW or rc == SQLITE_DONE) return;
        return error.SqliteError;
    }

    fn execUnlocked(self: *Database, sql: []const u8) !void {
        const sql_z = try self.allocator.dupeZ(u8, sql);
        defer self.allocator.free(sql_z);
        try self.check(sqlite3_exec(self.handle, sql_z.ptr, null, null, null));
    }

    fn exec(self: *Database, sql: []const u8) !void {
        self.mutex.lock();
        defer self.mutex.unlock();
        try self.execUnlocked(sql);
    }

    fn prepareUnlocked(self: *Database, sql: []const u8) !?*sqlite3_stmt {
        var stmt: ?*sqlite3_stmt = null;
        try self.check(sqlite3_prepare_v2(self.handle, sql.ptr, @intCast(sql.len), &stmt, null));
        return stmt;
    }

    fn finalize(self: *Database, stmt: ?*sqlite3_stmt) void {
        _ = self;
        _ = sqlite3_finalize(stmt);
    }

    fn bindText(self: *Database, stmt: ?*sqlite3_stmt, index: c_int, value: []const u8) !void {
        try self.check(sqlite3_bind_text(stmt, index, value.ptr, @intCast(value.len), @ptrFromInt(@as(usize, @bitCast(SQLITE_TRANSIENT)))));
    }

    fn bindInt64(self: *Database, stmt: ?*sqlite3_stmt, index: c_int, value: i64) !void {
        try self.check(sqlite3_bind_int64(stmt, index, value));
    }

    fn bindInt(self: *Database, stmt: ?*sqlite3_stmt, index: c_int, value: c_int) !void {
        try self.check(sqlite3_bind_int(stmt, index, value));
    }

    fn bindDouble(self: *Database, stmt: ?*sqlite3_stmt, index: c_int, value: f64) !void {
        try self.check(sqlite3_bind_double(stmt, index, value));
    }

    fn bindNull(self: *Database, stmt: ?*sqlite3_stmt, index: c_int) !void {
        try self.check(sqlite3_bind_null(stmt, index));
    }

    fn columnText(self: *Database, stmt: ?*sqlite3_stmt, index: c_int) ![]u8 {
        const ptr = sqlite3_column_text(stmt, index);
        if (ptr == null) return self.allocator.dupe(u8, "");
        const len_i = sqlite3_column_bytes(stmt, index);
        if (len_i < 0) return error.SqliteError;
        const len: usize = @intCast(len_i);
        return self.allocator.dupe(u8, ptr[0..len]);
    }

    fn initSchema(self: *Database) !void {
        try self.exec(
            \\PRAGMA journal_mode=WAL;
            \\PRAGMA synchronous=NORMAL;
            \\PRAGMA foreign_keys=ON;
            \\CREATE TABLE IF NOT EXISTS tenants(id TEXT PRIMARY KEY,name TEXT NOT NULL,created_at INTEGER NOT NULL);
            \\CREATE TABLE IF NOT EXISTS runs(id TEXT PRIMARY KEY,tenant_id TEXT NOT NULL,procedure_json TEXT NOT NULL,status TEXT NOT NULL,created_at INTEGER NOT NULL,updated_at INTEGER NOT NULL,step INTEGER NOT NULL DEFAULT 0,latest_observation_json TEXT NOT NULL DEFAULT '{"type":"start"}',terminal_result_json TEXT,error TEXT,token_budget_used INTEGER NOT NULL DEFAULT 0,token_budget_limit INTEGER NOT NULL DEFAULT 10000000,finalized INTEGER NOT NULL DEFAULT 0,FOREIGN KEY(tenant_id) REFERENCES tenants(id));
            \\CREATE INDEX IF NOT EXISTS runs_tenant_status_idx ON runs(tenant_id,status);
            \\CREATE TABLE IF NOT EXISTS run_state(run_id TEXT NOT NULL,key TEXT NOT NULL,value_json TEXT NOT NULL,updated_at INTEGER NOT NULL,PRIMARY KEY(run_id,key),FOREIGN KEY(run_id) REFERENCES runs(id) ON DELETE CASCADE);
            \\CREATE TABLE IF NOT EXISTS checkpoints(id INTEGER PRIMARY KEY AUTOINCREMENT,run_id TEXT NOT NULL,step INTEGER NOT NULL,state_json TEXT NOT NULL,patch_json TEXT NOT NULL,observation_json TEXT NOT NULL,action_json TEXT NOT NULL,created_at INTEGER NOT NULL,FOREIGN KEY(run_id) REFERENCES runs(id) ON DELETE CASCADE);
            \\CREATE TABLE IF NOT EXISTS events(id INTEGER PRIMARY KEY AUTOINCREMENT,tenant_id TEXT NOT NULL,run_id TEXT,step INTEGER NOT NULL,type TEXT NOT NULL,payload_json TEXT NOT NULL,created_at INTEGER NOT NULL);
            \\CREATE INDEX IF NOT EXISTS events_run_id_idx ON events(run_id,id);
            \\CREATE TABLE IF NOT EXISTS observation_queue(id INTEGER PRIMARY KEY AUTOINCREMENT,run_id TEXT NOT NULL,observation_json TEXT NOT NULL,consumed INTEGER NOT NULL DEFAULT 0,created_at INTEGER NOT NULL,FOREIGN KEY(run_id) REFERENCES runs(id) ON DELETE CASCADE);
            \\CREATE INDEX IF NOT EXISTS observation_queue_run_idx ON observation_queue(run_id,consumed,id);
            \\CREATE TABLE IF NOT EXISTS pending_actions(id INTEGER PRIMARY KEY AUTOINCREMENT,run_id TEXT NOT NULL,step INTEGER NOT NULL,action_json TEXT NOT NULL,status TEXT NOT NULL,result_json TEXT,created_at INTEGER NOT NULL,updated_at INTEGER NOT NULL,FOREIGN KEY(run_id) REFERENCES runs(id) ON DELETE CASCADE);
            \\CREATE INDEX IF NOT EXISTS pending_actions_run_idx ON pending_actions(run_id,status,id);
            \\CREATE TABLE IF NOT EXISTS cognition_frames(id INTEGER PRIMARY KEY AUTOINCREMENT,run_id TEXT NOT NULL,step INTEGER NOT NULL,frame_json TEXT NOT NULL,generated_at INTEGER NOT NULL,FOREIGN KEY(run_id) REFERENCES runs(id) ON DELETE CASCADE);
            \\CREATE INDEX IF NOT EXISTS cognition_frames_run_idx ON cognition_frames(run_id,id);
            \\CREATE TABLE IF NOT EXISTS skills(id TEXT PRIMARY KEY,tenant_id TEXT NOT NULL,name TEXT NOT NULL,description TEXT NOT NULL,trigger_json TEXT NOT NULL,procedure_json TEXT NOT NULL,embedding_json TEXT NOT NULL,version INTEGER NOT NULL,enabled INTEGER NOT NULL,success_count INTEGER NOT NULL DEFAULT 0,failure_count INTEGER NOT NULL DEFAULT 0,created_at INTEGER NOT NULL,updated_at INTEGER NOT NULL,FOREIGN KEY(tenant_id) REFERENCES tenants(id));
            \\CREATE INDEX IF NOT EXISTS skills_tenant_enabled_idx ON skills(tenant_id,enabled);
            \\CREATE VIRTUAL TABLE IF NOT EXISTS skill_fts USING fts5(id UNINDEXED,name,description,procedure);
            \\CREATE TABLE IF NOT EXISTS raw_traces(id INTEGER PRIMARY KEY AUTOINCREMENT,tenant_id TEXT NOT NULL,run_id TEXT NOT NULL,step INTEGER NOT NULL,initial_state_json TEXT NOT NULL,selected_skill_json TEXT NOT NULL,observation_json TEXT NOT NULL,model_envelope_json TEXT NOT NULL,state_patch_json TEXT NOT NULL,action_json TEXT NOT NULL,outcome_json TEXT NOT NULL,post_state_json TEXT NOT NULL,verifier_result_json TEXT NOT NULL,created_at INTEGER NOT NULL);
            \\CREATE TABLE IF NOT EXISTS knowledge_versions(id INTEGER PRIMARY KEY AUTOINCREMENT,tenant_id TEXT NOT NULL,content_markdown TEXT NOT NULL,git_commit TEXT NOT NULL,diff_text TEXT NOT NULL,created_at INTEGER NOT NULL);
            \\CREATE TABLE IF NOT EXISTS skill_patches(id TEXT PRIMARY KEY,tenant_id TEXT NOT NULL,source_trace_id INTEGER,status TEXT NOT NULL,patch_json TEXT NOT NULL,validation_report_json TEXT NOT NULL,created_at INTEGER NOT NULL,updated_at INTEGER NOT NULL);
            \\CREATE INDEX IF NOT EXISTS skill_patches_status_idx ON skill_patches(tenant_id,status);
            \\CREATE TABLE IF NOT EXISTS regression_tasks(id TEXT PRIMARY KEY,tenant_id TEXT NOT NULL,query_text TEXT NOT NULL,expected_skill_id TEXT NOT NULL,created_at INTEGER NOT NULL);
            \\CREATE TABLE IF NOT EXISTS distillation_examples(id INTEGER PRIMARY KEY AUTOINCREMENT,tenant_id TEXT NOT NULL,run_id TEXT NOT NULL,reflection_patch_json TEXT NOT NULL,clean_prompt_json TEXT NOT NULL,teacher_logprobs_json TEXT NOT NULL,student_logprobs_json TEXT NOT NULL,reverse_kl REAL NOT NULL,created_at INTEGER NOT NULL);
        );
    }

    fn ensureTenant(self: *Database, tenant_id: []const u8) !void {
        self.mutex.lock();
        defer self.mutex.unlock();
        const stmt = try self.prepareUnlocked("INSERT OR IGNORE INTO tenants(id,name,created_at) VALUES(?,?,?);");
        defer self.finalize(stmt);
        try self.bindText(stmt, 1, tenant_id);
        try self.bindText(stmt, 2, tenant_id);
        try self.bindInt64(stmt, 3, now());
        try self.checkStep(sqlite3_step(stmt));
    }

    fn createRun(self: *Database, tenant_id: []const u8, run_id: []const u8, procedure_json: []const u8, initial_observation_json: []const u8, token_budget_limit: i64) !void {
        self.mutex.lock();
        defer self.mutex.unlock();
        try self.execUnlocked("BEGIN IMMEDIATE;");
        var committed = false;
        defer if (!committed) self.execUnlocked("ROLLBACK;") catch {};
        const stmt = try self.prepareUnlocked("INSERT INTO runs(id,tenant_id,procedure_json,status,created_at,updated_at,step,latest_observation_json,token_budget_limit) VALUES(?,?,?,?,?,?,?,?,?);");
        defer self.finalize(stmt);
        try self.bindText(stmt, 1, run_id);
        try self.bindText(stmt, 2, tenant_id);
        try self.bindText(stmt, 3, procedure_json);
        try self.bindText(stmt, 4, "running");
        try self.bindInt64(stmt, 5, now());
        try self.bindInt64(stmt, 6, now());
        try self.bindInt64(stmt, 7, 0);
        try self.bindText(stmt, 8, initial_observation_json);
        try self.bindInt64(stmt, 9, token_budget_limit);
        try self.checkStep(sqlite3_step(stmt));

        const initial_state = [_]struct { key: []const u8, value: []const u8 }{
            .{ .key = "goal", .value = "\"\"" },
            .{ .key = "status", .value = "\"planning\"" },
            .{ .key = "progress", .value = "0" },
            .{ .key = "subgoals", .value = "[]" },
            .{ .key = "completed", .value = "[]" },
            .{ .key = "blockers", .value = "[]" },
            .{ .key = "facts", .value = "{}" },
            .{ .key = "constraints", .value = "[]" },
            .{ .key = "artifacts", .value = "{}" },
            .{ .key = "verified_progress", .value = "[]" },
            .{ .key = "unresolved_dependencies", .value = "[]" },
            .{ .key = "environment_constraints", .value = "[]" },
            .{ .key = "active_skill_ids", .value = "[]" },
            .{ .key = "completion", .value = "{\"done\":false,\"reason\":\"\"}" },
            .{ .key = "step_status", .value = "\"initialized\"" },
        };
        for (initial_state) |entry| {
            const istmt = try self.prepareUnlocked("INSERT INTO run_state(run_id,key,value_json,updated_at) VALUES(?,?,?,?);");
            defer self.finalize(istmt);
            try self.bindText(istmt, 1, run_id);
            try self.bindText(istmt, 2, entry.key);
            try self.bindText(istmt, 3, entry.value);
            try self.bindInt64(istmt, 4, now());
            try self.checkStep(sqlite3_step(istmt));
        }
        const state_json = try self.getStateJsonUnlocked(run_id);
        defer self.allocator.free(state_json);
        const cstmt = try self.prepareUnlocked("INSERT INTO checkpoints(run_id,step,state_json,patch_json,observation_json,action_json,created_at) VALUES(?,?,?,?,?,?,?);");
        defer self.finalize(cstmt);
        try self.bindText(cstmt, 1, run_id);
        try self.bindInt64(cstmt, 2, 0);
        try self.bindText(cstmt, 3, state_json);
        try self.bindText(cstmt, 4, "{}");
        try self.bindText(cstmt, 5, initial_observation_json);
        try self.bindText(cstmt, 6, "{\"type\":\"none\"}");
        try self.bindInt64(cstmt, 7, now());
        try self.checkStep(sqlite3_step(cstmt));
        try self.execUnlocked("COMMIT;");
        committed = true;
    }

    fn getRun(self: *Database, run_id: []const u8) !RunRecord {
        self.mutex.lock();
        defer self.mutex.unlock();
        const stmt = try self.prepareUnlocked("SELECT id,tenant_id,procedure_json,status,step,latest_observation_json,token_budget_used,token_budget_limit FROM runs WHERE id=?;");
        defer self.finalize(stmt);
        try self.bindText(stmt, 1, run_id);
        const rc = sqlite3_step(stmt);
        try self.checkStep(rc);
        if (rc != SQLITE_ROW) return error.NotFound;
        return RunRecord{
            .id = try self.columnText(stmt, 0),
            .tenant_id = try self.columnText(stmt, 1),
            .procedure_json = try self.columnText(stmt, 2),
            .status = try self.columnText(stmt, 3),
            .step = sqlite3_column_int64(stmt, 4),
            .latest_observation_json = try self.columnText(stmt, 5),
            .token_budget_used = sqlite3_column_int64(stmt, 6),
            .token_budget_limit = sqlite3_column_int64(stmt, 7),
        };
    }

    fn getRunStatus(self: *Database, run_id: []const u8) ![]u8 {
        self.mutex.lock();
        defer self.mutex.unlock();
        const stmt = try self.prepareUnlocked("SELECT status FROM runs WHERE id=?;");
        defer self.finalize(stmt);
        try self.bindText(stmt, 1, run_id);
        const rc = sqlite3_step(stmt);
        try self.checkStep(rc);
        if (rc != SQLITE_ROW) return error.NotFound;
        return self.columnText(stmt, 0);
    }

    fn markRunStatus(self: *Database, run_id: []const u8, status: []const u8, error_json: ?[]const u8) !void {
        self.mutex.lock();
        defer self.mutex.unlock();
        const stmt = try self.prepareUnlocked("UPDATE runs SET status=?,error=?,updated_at=? WHERE id=?;");
        defer self.finalize(stmt);
        try self.bindText(stmt, 1, status);
        if (error_json) |e| try self.bindText(stmt, 2, e) else try self.bindNull(stmt, 2);
        try self.bindInt64(stmt, 3, now());
        try self.bindText(stmt, 4, run_id);
        try self.checkStep(sqlite3_step(stmt));
    }

    fn markCompleted(self: *Database, run_id: []const u8, result_json: []const u8) !void {
        self.mutex.lock();
        defer self.mutex.unlock();
        const stmt = try self.prepareUnlocked("UPDATE runs SET status='completed',terminal_result_json=?,updated_at=? WHERE id=? AND status IN ('running','queued');");
        defer self.finalize(stmt);
        try self.bindText(stmt, 1, result_json);
        try self.bindInt64(stmt, 2, now());
        try self.bindText(stmt, 3, run_id);
        try self.checkStep(sqlite3_step(stmt));
    }

    fn claimFinalization(self: *Database, run_id: []const u8) !bool {
        self.mutex.lock();
        defer self.mutex.unlock();
        const stmt = try self.prepareUnlocked("UPDATE runs SET finalized=1,updated_at=? WHERE id=? AND finalized=0 AND status IN ('completed','failed','stopped');");
        defer self.finalize(stmt);
        try self.bindInt64(stmt, 1, now());
        try self.bindText(stmt, 2, run_id);
        try self.checkStep(sqlite3_step(stmt));
        return sqlite3_changes(self.handle) > 0;
    }

    fn listActiveRunIds(self: *Database) ![][]u8 {
        self.mutex.lock();
        defer self.mutex.unlock();
        var list = std.ArrayList([]u8).init(self.allocator);
        errdefer {
            for (list.items) |id| self.allocator.free(id);
            list.deinit();
        }
        const stmt = try self.prepareUnlocked("SELECT id FROM runs WHERE status IN ('queued','running');");
        defer self.finalize(stmt);
        while (true) {
            const rc = sqlite3_step(stmt);
            try self.checkStep(rc);
            if (rc == SQLITE_DONE) break;
            const id = try self.columnText(stmt, 0);
            errdefer self.allocator.free(id);
            try list.append(id);
        }
        return list.toOwnedSlice();
    }

    fn listRunsJson(self: *Database, tenant_id: []const u8) ![]u8 {
        self.mutex.lock();
        defer self.mutex.unlock();
        var out = std.ArrayList(u8).init(self.allocator);
        const w = out.writer();
        try w.writeAll("[");
        var first = true;
        const stmt = try self.prepareUnlocked("SELECT id,status,step,created_at,updated_at,terminal_result_json,error,token_budget_used,token_budget_limit FROM runs WHERE tenant_id=? ORDER BY created_at DESC LIMIT 200;");
        defer self.finalize(stmt);
        try self.bindText(stmt, 1, tenant_id);
        while (true) {
            const rc = sqlite3_step(stmt);
            try self.checkStep(rc);
            if (rc == SQLITE_DONE) break;
            if (!first) try w.writeAll(",");
            first = false;
            try w.writeAll("{\"id\":");
            const id = try self.columnText(stmt, 0);
            defer self.allocator.free(id);
            try writeJsonString(w, id);
            try w.writeAll(",\"status\":");
            const status = try self.columnText(stmt, 1);
            defer self.allocator.free(status);
            try writeJsonString(w, status);
            try w.print(",\"step\":{},\"created_at\":{},\"updated_at\":{},\"terminal_result\":", .{ sqlite3_column_int64(stmt, 2), sqlite3_column_int64(stmt, 3), sqlite3_column_int64(stmt, 4) });
            const terminal = try self.columnText(stmt, 5);
            defer self.allocator.free(terminal);
            if (terminal.len == 0) try w.writeAll("null") else try w.writeAll(terminal);
            try w.writeAll(",\"error\":");
            const err = try self.columnText(stmt, 6);
            defer self.allocator.free(err);
            if (err.len == 0) try w.writeAll("null") else try w.writeAll(err);
            try w.print(",\"token_budget_used\":{},\"token_budget_limit\":{}}}", .{ sqlite3_column_int64(stmt, 7), sqlite3_column_int64(stmt, 8) });
        }
        try w.writeAll("]");
        return out.toOwnedSlice();
    }

    fn runJson(self: *Database, tenant_id: []const u8, run_id: []const u8) ![]u8 {
        var run = try self.getRun(run_id);
        defer run.deinit(self.allocator);
        if (!std.mem.eql(u8, run.tenant_id, tenant_id)) return error.NotFound;
        const state = try self.getStateJson(run_id);
        defer self.allocator.free(state);
        var out = std.ArrayList(u8).init(self.allocator);
        const w = out.writer();
        try w.writeAll("{\"id\":");
        try writeJsonString(w, run.id);
        try w.writeAll(",\"tenant_id\":");
        try writeJsonString(w, run.tenant_id);
        try w.writeAll(",\"status\":");
        try writeJsonString(w, run.status);
        try w.print(",\"step\":{},\"procedure\":", .{run.step});
        try w.writeAll(run.procedure_json);
        try w.writeAll(",\"state\":");
        try w.writeAll(state);
        try w.writeAll(",\"latest_observation\":");
        try w.writeAll(run.latest_observation_json);
        try w.print(",\"token_budget_used\":{},\"token_budget_limit\":{}}}", .{ run.token_budget_used, run.token_budget_limit });
        return out.toOwnedSlice();
    }

    fn getStateJson(self: *Database, run_id: []const u8) ![]u8 {
        self.mutex.lock();
        defer self.mutex.unlock();
        return self.getStateJsonUnlocked(run_id);
    }

    fn getStateJsonUnlocked(self: *Database, run_id: []const u8) ![]u8 {
        var out = std.ArrayList(u8).init(self.allocator);
        const w = out.writer();
        try w.writeAll("{");
        var first = true;
        const stmt = try self.prepareUnlocked("SELECT key,value_json FROM run_state WHERE run_id=? ORDER BY key ASC;");
        defer self.finalize(stmt);
        try self.bindText(stmt, 1, run_id);
        while (true) {
            const rc = sqlite3_step(stmt);
            try self.checkStep(rc);
            if (rc == SQLITE_DONE) break;
            if (!first) try w.writeAll(",");
            first = false;
            const key = try self.columnText(stmt, 0);
            defer self.allocator.free(key);
            const value = try self.columnText(stmt, 1);
            defer self.allocator.free(value);
            try writeJsonString(w, key);
            try w.writeAll(":");
            try w.writeAll(value);
        }
        try w.writeAll("}");
        return out.toOwnedSlice();
    }

    fn applyPatchAndCheckpoint(self: *Database, run_id: []const u8, next_step: i64, patch_json: []const u8, observation_json: []const u8, action_json: []const u8, max_state_bytes: usize) ![]u8 {
        var parsed = try std.json.parseFromSlice(std.json.Value, self.allocator, patch_json, .{});
        defer parsed.deinit();
        try validatePatchValue(parsed.value);
        self.mutex.lock();
        defer self.mutex.unlock();
        try self.execUnlocked("BEGIN IMMEDIATE;");
        var committed = false;
        defer if (!committed) self.execUnlocked("ROLLBACK;") catch {};
        switch (parsed.value) {
            .object => |obj| {
                var it = obj.iterator();
                while (it.next()) |entry| {
                    const key = entry.key_ptr.*;
                    if (containsBannedKey(key)) return error.InvalidStatePatch;
                    switch (entry.value_ptr.*) {
                        .null => {
                            const stmt = try self.prepareUnlocked("DELETE FROM run_state WHERE run_id=? AND key=?;");
                            defer self.finalize(stmt);
                            try self.bindText(stmt, 1, run_id);
                            try self.bindText(stmt, 2, key);
                            try self.checkStep(sqlite3_step(stmt));
                        },
                        else => {
                            const value_json = try jsonValueToOwned(self.allocator, entry.value_ptr.*);
                            defer self.allocator.free(value_json);
                            const stmt = try self.prepareUnlocked("INSERT INTO run_state(run_id,key,value_json,updated_at) VALUES(?,?,?,?) ON CONFLICT(run_id,key) DO UPDATE SET value_json=excluded.value_json,updated_at=excluded.updated_at;");
                            defer self.finalize(stmt);
                            try self.bindText(stmt, 1, run_id);
                            try self.bindText(stmt, 2, key);
                            try self.bindText(stmt, 3, value_json);
                            try self.bindInt64(stmt, 4, now());
                            try self.checkStep(sqlite3_step(stmt));
                        },
                    }
                }
            },
            else => return error.InvalidStatePatch,
        }
        const state_json = try self.getStateJsonUnlocked(run_id);
        if (state_json.len > max_state_bytes) {
            self.allocator.free(state_json);
            return error.StateTooLarge;
        }
        const ustmt = try self.prepareUnlocked("UPDATE runs SET step=?,updated_at=? WHERE id=?;");
        defer self.finalize(ustmt);
        try self.bindInt64(ustmt, 1, next_step);
        try self.bindInt64(ustmt, 2, now());
        try self.bindText(ustmt, 3, run_id);
        try self.checkStep(sqlite3_step(ustmt));
        const cstmt = try self.prepareUnlocked("INSERT INTO checkpoints(run_id,step,state_json,patch_json,observation_json,action_json,created_at) VALUES(?,?,?,?,?,?,?);");
        defer self.finalize(cstmt);
        try self.bindText(cstmt, 1, run_id);
        try self.bindInt64(cstmt, 2, next_step);
        try self.bindText(cstmt, 3, state_json);
        try self.bindText(cstmt, 4, patch_json);
        try self.bindText(cstmt, 5, observation_json);
        try self.bindText(cstmt, 6, action_json);
        try self.bindInt64(cstmt, 7, now());
        try self.checkStep(sqlite3_step(cstmt));
        try self.execUnlocked("COMMIT;");
        committed = true;
        return state_json;
    }

    fn setRunStateKey(self: *Database, run_id: []const u8, key: []const u8, value_json: []const u8) !void {
        self.mutex.lock();
        defer self.mutex.unlock();
        const stmt = try self.prepareUnlocked("INSERT INTO run_state(run_id,key,value_json,updated_at) VALUES(?,?,?,?) ON CONFLICT(run_id,key) DO UPDATE SET value_json=excluded.value_json,updated_at=excluded.updated_at;");
        defer self.finalize(stmt);
        try self.bindText(stmt, 1, run_id);
        try self.bindText(stmt, 2, key);
        try self.bindText(stmt, 3, value_json);
        try self.bindInt64(stmt, 4, now());
        try self.checkStep(sqlite3_step(stmt));
    }

    fn insertEvent(self: *Database, tenant_id: []const u8, run_id: ?[]const u8, step: i64, typ: []const u8, payload_json: []const u8) !i64 {
        self.mutex.lock();
        defer self.mutex.unlock();
        const stmt = try self.prepareUnlocked("INSERT INTO events(tenant_id,run_id,step,type,payload_json,created_at) VALUES(?,?,?,?,?,?);");
        defer self.finalize(stmt);
        try self.bindText(stmt, 1, tenant_id);
        if (run_id) |rid| try self.bindText(stmt, 2, rid) else try self.bindNull(stmt, 2);
        try self.bindInt64(stmt, 3, step);
        try self.bindText(stmt, 4, typ);
        try self.bindText(stmt, 5, payload_json);
        try self.bindInt64(stmt, 6, now());
        try self.checkStep(sqlite3_step(stmt));
        return sqlite3_last_insert_rowid(self.handle);
    }

    fn enqueueObservation(self: *Database, run_id: []const u8, observation_json: []const u8) !void {
        self.mutex.lock();
        defer self.mutex.unlock();
        const stmt = try self.prepareUnlocked("INSERT INTO observation_queue(run_id,observation_json,created_at) VALUES(?,?,?);");
        defer self.finalize(stmt);
        try self.bindText(stmt, 1, run_id);
        try self.bindText(stmt, 2, observation_json);
        try self.bindInt64(stmt, 3, now());
        try self.checkStep(sqlite3_step(stmt));
    }

    fn takeObservation(self: *Database, run_id: []const u8) ![]u8 {
        self.mutex.lock();
        defer self.mutex.unlock();
        try self.execUnlocked("BEGIN IMMEDIATE;");
        var committed = false;
        defer if (!committed) self.execUnlocked("ROLLBACK;") catch {};
        const stmt = try self.prepareUnlocked("SELECT id,observation_json FROM observation_queue WHERE run_id=? AND consumed=0 ORDER BY id ASC LIMIT 1;");
        defer self.finalize(stmt);
        try self.bindText(stmt, 1, run_id);
        const rc = sqlite3_step(stmt);
        try self.checkStep(rc);
        if (rc == SQLITE_ROW) {
            const obs_id = sqlite3_column_int64(stmt, 0);
            const obs = try self.columnText(stmt, 1);
            const ustmt = try self.prepareUnlocked("UPDATE observation_queue SET consumed=1 WHERE id=?;");
            defer self.finalize(ustmt);
            try self.bindInt64(ustmt, 1, obs_id);
            try self.checkStep(sqlite3_step(ustmt));
            const rstmt = try self.prepareUnlocked("UPDATE runs SET latest_observation_json=?,updated_at=? WHERE id=?;");
            defer self.finalize(rstmt);
            try self.bindText(rstmt, 1, obs);
            try self.bindInt64(rstmt, 2, now());
            try self.bindText(rstmt, 3, run_id);
            try self.checkStep(sqlite3_step(rstmt));
            try self.execUnlocked("COMMIT;");
            committed = true;
            return obs;
        }
        const rstmt = try self.prepareUnlocked("SELECT latest_observation_json FROM runs WHERE id=?;");
        defer self.finalize(rstmt);
        try self.bindText(rstmt, 1, run_id);
        const rrc = sqlite3_step(rstmt);
        try self.checkStep(rrc);
        if (rrc != SQLITE_ROW) return error.NotFound;
        const obs = try self.columnText(rstmt, 0);
        try self.execUnlocked("COMMIT;");
        committed = true;
        return obs;
    }

    fn updateLatestObservation(self: *Database, run_id: []const u8, observation_json: []const u8) !void {
        self.mutex.lock();
        defer self.mutex.unlock();
        const stmt = try self.prepareUnlocked("UPDATE runs SET latest_observation_json=?,updated_at=? WHERE id=?;");
        defer self.finalize(stmt);
        try self.bindText(stmt, 1, observation_json);
        try self.bindInt64(stmt, 2, now());
        try self.bindText(stmt, 3, run_id);
        try self.checkStep(sqlite3_step(stmt));
    }

    fn enqueueAction(self: *Database, run_id: []const u8, step: i64, action_json: []const u8) !i64 {
        self.mutex.lock();
        defer self.mutex.unlock();
        const stmt = try self.prepareUnlocked("INSERT INTO pending_actions(run_id,step,action_json,status,created_at,updated_at) VALUES(?,?,?,'pending',?,?);");
        defer self.finalize(stmt);
        try self.bindText(stmt, 1, run_id);
        try self.bindInt64(stmt, 2, step);
        try self.bindText(stmt, 3, action_json);
        try self.bindInt64(stmt, 4, now());
        try self.bindInt64(stmt, 5, now());
        try self.checkStep(sqlite3_step(stmt));
        return sqlite3_last_insert_rowid(self.handle);
    }

    fn hasOpenAction(self: *Database, run_id: []const u8) !bool {
        self.mutex.lock();
        defer self.mutex.unlock();
        const stmt = try self.prepareUnlocked("SELECT id FROM pending_actions WHERE run_id=? AND status IN ('pending','executing') LIMIT 1;");
        defer self.finalize(stmt);
        try self.bindText(stmt, 1, run_id);
        const rc = sqlite3_step(stmt);
        try self.checkStep(rc);
        return rc == SQLITE_ROW;
    }

    fn claimPendingAction(self: *Database, run_id: []const u8) !?ActionClaim {
        self.mutex.lock();
        defer self.mutex.unlock();
        try self.execUnlocked("BEGIN IMMEDIATE;");
        var committed = false;
        defer if (!committed) self.execUnlocked("ROLLBACK;") catch {};
        const stmt = try self.prepareUnlocked("SELECT id,run_id,step,action_json FROM pending_actions WHERE run_id=? AND status='pending' ORDER BY id ASC LIMIT 1;");
        defer self.finalize(stmt);
        try self.bindText(stmt, 1, run_id);
        const rc = sqlite3_step(stmt);
        try self.checkStep(rc);
        if (rc != SQLITE_ROW) {
            try self.execUnlocked("COMMIT;");
            committed = true;
            return null;
        }
        const id = sqlite3_column_int64(stmt, 0);
        const rid = try self.columnText(stmt, 1);
        const step = sqlite3_column_int64(stmt, 2);
        const action_json = try self.columnText(stmt, 3);
        const ustmt = try self.prepareUnlocked("UPDATE pending_actions SET status='executing',updated_at=? WHERE id=? AND status='pending';");
        defer self.finalize(ustmt);
        try self.bindInt64(ustmt, 1, now());
        try self.bindInt64(ustmt, 2, id);
        try self.checkStep(sqlite3_step(ustmt));
        try self.execUnlocked("COMMIT;");
        committed = true;
        return ActionClaim{ .id = id, .run_id = rid, .step = step, .action_json = action_json };
    }

    fn completeAction(self: *Database, action_id: i64, outcome_json: []const u8) !void {
        self.mutex.lock();
        defer self.mutex.unlock();
        const stmt = try self.prepareUnlocked("UPDATE pending_actions SET status='done',result_json=?,updated_at=? WHERE id=?;");
        defer self.finalize(stmt);
        try self.bindText(stmt, 1, outcome_json);
        try self.bindInt64(stmt, 2, now());
        try self.bindInt64(stmt, 3, action_id);
        try self.checkStep(sqlite3_step(stmt));
    }

    fn getActionResult(self: *Database, action_id: i64) !?[]u8 {
        self.mutex.lock();
        defer self.mutex.unlock();
        const stmt = try self.prepareUnlocked("SELECT result_json FROM pending_actions WHERE id=? AND status='done';");
        defer self.finalize(stmt);
        try self.bindInt64(stmt, 1, action_id);
        const rc = sqlite3_step(stmt);
        try self.checkStep(rc);
        if (rc != SQLITE_ROW) return null;
        return try self.columnText(stmt, 0);
    }

    fn insertCognitionFrame(self: *Database, run_id: []const u8, step: i64, frame_json: []const u8) !void {
        self.mutex.lock();
        defer self.mutex.unlock();
        const stmt = try self.prepareUnlocked("INSERT INTO cognition_frames(run_id,step,frame_json,generated_at) VALUES(?,?,?,?);");
        defer self.finalize(stmt);
        try self.bindText(stmt, 1, run_id);
        try self.bindInt64(stmt, 2, step);
        try self.bindText(stmt, 3, frame_json);
        try self.bindInt64(stmt, 4, nowMillis());
        try self.checkStep(sqlite3_step(stmt));
    }

    fn latestCognitionGeneratedAt(self: *Database, run_id: []const u8) !i64 {
        self.mutex.lock();
        defer self.mutex.unlock();
        const stmt = try self.prepareUnlocked("SELECT generated_at FROM cognition_frames WHERE run_id=? ORDER BY id DESC LIMIT 1;");
        defer self.finalize(stmt);
        try self.bindText(stmt, 1, run_id);
        const rc = sqlite3_step(stmt);
        try self.checkStep(rc);
        if (rc != SQLITE_ROW) return nowMillis();
        return sqlite3_column_int64(stmt, 0);
    }

    fn addRawTrace(self: *Database, tenant_id: []const u8, run_id: []const u8, step: i64, initial_state_json: []const u8, selected_skill_json: []const u8, observation_json: []const u8, model_envelope_json: []const u8, state_patch_json: []const u8, action_json: []const u8, outcome_json: []const u8, post_state_json: []const u8, verifier_result_json: []const u8) !void {
        self.mutex.lock();
        defer self.mutex.unlock();
        const stmt = try self.prepareUnlocked("INSERT INTO raw_traces(tenant_id,run_id,step,initial_state_json,selected_skill_json,observation_json,model_envelope_json,state_patch_json,action_json,outcome_json,post_state_json,verifier_result_json,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?);");
        defer self.finalize(stmt);
        try self.bindText(stmt, 1, tenant_id);
        try self.bindText(stmt, 2, run_id);
        try self.bindInt64(stmt, 3, step);
        try self.bindText(stmt, 4, initial_state_json);
        try self.bindText(stmt, 5, selected_skill_json);
        try self.bindText(stmt, 6, observation_json);
        try self.bindText(stmt, 7, model_envelope_json);
        try self.bindText(stmt, 8, state_patch_json);
        try self.bindText(stmt, 9, action_json);
        try self.bindText(stmt, 10, outcome_json);
        try self.bindText(stmt, 11, post_state_json);
        try self.bindText(stmt, 12, verifier_result_json);
        try self.bindInt64(stmt, 13, now());
        try self.checkStep(sqlite3_step(stmt));
    }

    fn loadEnabledSkills(self: *Database, tenant_id: []const u8) ![]SkillRecord {
        self.mutex.lock();
        defer self.mutex.unlock();
        var list = std.ArrayList(SkillRecord).init(self.allocator);
        errdefer {
            for (list.items) |*record| record.deinit(self.allocator);
            list.deinit();
        }
        const stmt = try self.prepareUnlocked("SELECT id,name,description,trigger_json,procedure_json,embedding_json FROM skills WHERE tenant_id=? AND enabled=1 ORDER BY updated_at DESC LIMIT 1000;");
        defer self.finalize(stmt);
        try self.bindText(stmt, 1, tenant_id);
        while (true) {
            const rc = sqlite3_step(stmt);
            try self.checkStep(rc);
            if (rc == SQLITE_DONE) break;
            const id = try self.columnText(stmt, 0);
            errdefer self.allocator.free(id);
            const name = try self.columnText(stmt, 1);
            errdefer self.allocator.free(name);
            const description = try self.columnText(stmt, 2);
            errdefer self.allocator.free(description);
            const trigger_json = try self.columnText(stmt, 3);
            errdefer self.allocator.free(trigger_json);
            const procedure_json = try self.columnText(stmt, 4);
            errdefer self.allocator.free(procedure_json);
            const embedding_json = try self.columnText(stmt, 5);
            errdefer self.allocator.free(embedding_json);
            try list.append(SkillRecord{
                .id = id,
                .name = name,
                .description = description,
                .trigger_json = trigger_json,
                .procedure_json = procedure_json,
                .embedding_json = embedding_json,
                .vector_score = 0,
                .sparse_rank = std.math.maxInt(usize),
                .rrf_score = 0,
            });
        }
        return list.toOwnedSlice();
    }

    fn searchSkillFtsIds(self: *Database, tenant_id: []const u8, fts_query: []const u8) ![][]u8 {
        self.mutex.lock();
        defer self.mutex.unlock();
        var ids = std.ArrayList([]u8).init(self.allocator);
        errdefer {
            for (ids.items) |id| self.allocator.free(id);
            ids.deinit();
        }
        const stmt = try self.prepareUnlocked("SELECT skills.id FROM skill_fts JOIN skills ON skills.id=skill_fts.id WHERE skill_fts MATCH ? AND skills.tenant_id=? AND skills.enabled=1 ORDER BY bm25(skill_fts) LIMIT 20;");
        defer self.finalize(stmt);
        try self.bindText(stmt, 1, fts_query);
        try self.bindText(stmt, 2, tenant_id);
        while (true) {
            const rc = sqlite3_step(stmt);
            try self.checkStep(rc);
            if (rc == SQLITE_DONE) break;
            const id = try self.columnText(stmt, 0);
            errdefer self.allocator.free(id);
            try ids.append(id);
        }
        return ids.toOwnedSlice();
    }

    fn addSkillFromFields(self: *Database, tenant_id: []const u8, id: []const u8, name: []const u8, description: []const u8, trigger_json: []const u8, procedure_json: []const u8, tokens_json: []const u8, enabled: bool) !void {
        self.mutex.lock();
        defer self.mutex.unlock();
        try self.execUnlocked("BEGIN IMMEDIATE;");
        var committed = false;
        defer if (!committed) self.execUnlocked("ROLLBACK;") catch {};
        const stmt = try self.prepareUnlocked("INSERT INTO skills(id,tenant_id,name,description,trigger_json,procedure_json,embedding_json,version,enabled,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET name=excluded.name,description=excluded.description,trigger_json=excluded.trigger_json,procedure_json=excluded.procedure_json,embedding_json=excluded.embedding_json,version=skills.version+1,enabled=excluded.enabled,updated_at=excluded.updated_at;");
        defer self.finalize(stmt);
        try self.bindText(stmt, 1, id);
        try self.bindText(stmt, 2, tenant_id);
        try self.bindText(stmt, 3, name);
        try self.bindText(stmt, 4, description);
        try self.bindText(stmt, 5, trigger_json);
        try self.bindText(stmt, 6, procedure_json);
        try self.bindText(stmt, 7, tokens_json);
        try self.bindInt64(stmt, 8, 1);
        try self.bindInt(stmt, 9, if (enabled) 1 else 0);
        try self.bindInt64(stmt, 10, now());
        try self.bindInt64(stmt, 11, now());
        try self.checkStep(sqlite3_step(stmt));
        const dstmt = try self.prepareUnlocked("DELETE FROM skill_fts WHERE id=?;");
        defer self.finalize(dstmt);
        try self.bindText(dstmt, 1, id);
        try self.checkStep(sqlite3_step(dstmt));
        const fstmt = try self.prepareUnlocked("INSERT INTO skill_fts(id,name,description,procedure) VALUES(?,?,?,?);");
        defer self.finalize(fstmt);
        try self.bindText(fstmt, 1, id);
        try self.bindText(fstmt, 2, name);
        try self.bindText(fstmt, 3, description);
        try self.bindText(fstmt, 4, procedure_json);
        try self.checkStep(sqlite3_step(fstmt));
        try self.execUnlocked("COMMIT;");
        committed = true;
    }

    fn incrementTokenUsage(self: *Database, run_id: []const u8, amount: i64) !void {
        self.mutex.lock();
        defer self.mutex.unlock();
        const stmt = try self.prepareUnlocked("UPDATE runs SET token_budget_used=token_budget_used+?,updated_at=? WHERE id=?;");
        defer self.finalize(stmt);
        try self.bindInt64(stmt, 1, amount);
        try self.bindInt64(stmt, 2, now());
        try self.bindText(stmt, 3, run_id);
        try self.checkStep(sqlite3_step(stmt));
    }

    fn eventsSinceJson(self: *Database, tenant_id: []const u8, run_id: []const u8, last_id: i64) ![]u8 {
        self.mutex.lock();
        defer self.mutex.unlock();
        var out = std.ArrayList(u8).init(self.allocator);
        const w = out.writer();
        try w.writeAll("[");
        var first = true;
        const stmt = try self.prepareUnlocked("SELECT id,step,type,payload_json,created_at FROM events WHERE tenant_id=? AND run_id=? AND id>? ORDER BY id ASC LIMIT 200;");
        defer self.finalize(stmt);
        try self.bindText(stmt, 1, tenant_id);
        try self.bindText(stmt, 2, run_id);
        try self.bindInt64(stmt, 3, last_id);
        while (true) {
            const rc = sqlite3_step(stmt);
            try self.checkStep(rc);
            if (rc == SQLITE_DONE) break;
            if (!first) try w.writeAll(",");
            first = false;
            const typ = try self.columnText(stmt, 2);
            defer self.allocator.free(typ);
            const payload = try self.columnText(stmt, 3);
            defer self.allocator.free(payload);
            try w.print("{{\"id\":{},\"step\":{},\"type\":", .{ sqlite3_column_int64(stmt, 0), sqlite3_column_int64(stmt, 1) });
            try writeJsonString(w, typ);
            try w.writeAll(",\"payload\":");
            try w.writeAll(if (payload.len == 0) "null" else payload);
            try w.print(",\"created_at\":{}}}", .{sqlite3_column_int64(stmt, 4)});
        }
        try w.writeAll("]");
        return out.toOwnedSlice();
    }

    fn insertSkillPatch(self: *Database, tenant_id: []const u8, patch_id: []const u8, source_trace_id: ?i64, status: []const u8, patch_json: []const u8, validation_report_json: []const u8) !void {
        self.mutex.lock();
        defer self.mutex.unlock();
        const stmt = try self.prepareUnlocked("INSERT INTO skill_patches(id,tenant_id,source_trace_id,status,patch_json,validation_report_json,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET status=excluded.status,validation_report_json=excluded.validation_report_json,updated_at=excluded.updated_at;");
        defer self.finalize(stmt);
        try self.bindText(stmt, 1, patch_id);
        try self.bindText(stmt, 2, tenant_id);
        if (source_trace_id) |sid| try self.bindInt64(stmt, 3, sid) else try self.bindNull(stmt, 3);
        try self.bindText(stmt, 4, status);
        try self.bindText(stmt, 5, patch_json);
        try self.bindText(stmt, 6, validation_report_json);
        try self.bindInt64(stmt, 7, now());
        try self.bindInt64(stmt, 8, now());
        try self.checkStep(sqlite3_step(stmt));
    }

    fn insertDistillationExample(self: *Database, tenant_id: []const u8, run_id: []const u8, reflection_patch_json: []const u8, clean_prompt_json: []const u8, teacher_logprobs_json: []const u8, student_logprobs_json: []const u8, reverse_kl: f64) !void {
        self.mutex.lock();
        defer self.mutex.unlock();
        const stmt = try self.prepareUnlocked("INSERT INTO distillation_examples(tenant_id,run_id,reflection_patch_json,clean_prompt_json,teacher_logprobs_json,student_logprobs_json,reverse_kl,created_at) VALUES(?,?,?,?,?,?,?,?);");
        defer self.finalize(stmt);
        try self.bindText(stmt, 1, tenant_id);
        try self.bindText(stmt, 2, run_id);
        try self.bindText(stmt, 3, reflection_patch_json);
        try self.bindText(stmt, 4, clean_prompt_json);
        try self.bindText(stmt, 5, teacher_logprobs_json);
        try self.bindText(stmt, 6, student_logprobs_json);
        try self.bindDouble(stmt, 7, reverse_kl);
        try self.bindInt64(stmt, 8, now());
        try self.checkStep(sqlite3_step(stmt));
    }

    fn buildKnowledgeMarkdown(self: *Database, tenant_id: []const u8) ![]u8 {
        self.mutex.lock();
        defer self.mutex.unlock();
        var out = std.ArrayList(u8).init(self.allocator);
        const w = out.writer();
        try w.writeAll("# Autonomous Agent Playbook\n\n## Verified Skills\n\n");
        const stmt = try self.prepareUnlocked("SELECT name,description,procedure_json,success_count,failure_count,version FROM skills WHERE tenant_id=? AND enabled=1 ORDER BY updated_at DESC LIMIT 200;");
        defer self.finalize(stmt);
        try self.bindText(stmt, 1, tenant_id);
        while (true) {
            const rc = sqlite3_step(stmt);
            try self.checkStep(rc);
            if (rc == SQLITE_DONE) break;
            const name = try self.columnText(stmt, 0);
            defer self.allocator.free(name);
            const description = try self.columnText(stmt, 1);
            defer self.allocator.free(description);
            const procedure = try self.columnText(stmt, 2);
            defer self.allocator.free(procedure);
            try w.print("### {s}\n\n{s}\n\nVersion: {}\nSuccesses: {}\nFailures: {}\n\nProcedure:\n\n```json\n{s}\n```\n\n", .{ name, description, sqlite3_column_int64(stmt, 5), sqlite3_column_int64(stmt, 3), sqlite3_column_int64(stmt, 4), procedure });
        }
        return out.toOwnedSlice();
    }

    fn insertKnowledgeVersion(self: *Database, tenant_id: []const u8, content: []const u8, commit: []const u8, diff: []const u8) !void {
        self.mutex.lock();
        defer self.mutex.unlock();
        const stmt = try self.prepareUnlocked("INSERT INTO knowledge_versions(tenant_id,content_markdown,git_commit,diff_text,created_at) VALUES(?,?,?,?,?);");
        defer self.finalize(stmt);
        try self.bindText(stmt, 1, tenant_id);
        try self.bindText(stmt, 2, content);
        try self.bindText(stmt, 3, commit);
        try self.bindText(stmt, 4, diff);
        try self.bindInt64(stmt, 5, now());
        try self.checkStep(sqlite3_step(stmt));
    }
};

const App = struct {
    allocator: Allocator,
    config: Config,
    db: *Database,
    fs_mutex: std.Thread.Mutex,
    retrieval_mutex: std.Thread.Mutex,
    graph_mutex: std.Thread.Mutex,
    mgt: MGT,
    ssi: SSI,
    ranker: Ranker,
    kernel: ChaosCoreKernel,
    crev: CREVPipeline,
};

const ExprEval = struct {
    src: []const u8,
    pos: usize,

    pub fn skipWs(self: *ExprEval) void {
        while (self.pos < self.src.len and (self.src[self.pos] == ' ' or self.src[self.pos] == '\t' or self.src[self.pos] == '\r' or self.src[self.pos] == '\n')) self.pos += 1;
    }

    pub fn parseExpr(self: *ExprEval) !f64 {
        var left = try self.parseTerm();
        while (true) {
            self.skipWs();
            if (self.pos >= self.src.len) break;
            const ch = self.src[self.pos];
            if (ch == '+') {
                self.pos += 1;
                const right = try self.parseTerm();
                left += right;
            } else if (ch == '-') {
                self.pos += 1;
                const right = try self.parseTerm();
                left -= right;
            } else break;
        }
        return left;
    }

    fn parseTerm(self: *ExprEval) !f64 {
        var left = try self.parseUnary();
        while (true) {
            self.skipWs();
            if (self.pos >= self.src.len) break;
            const ch = self.src[self.pos];
            if (ch == '*') {
                if (self.pos + 1 < self.src.len and self.src[self.pos + 1] == '*') break;
                self.pos += 1;
                const right = try self.parseUnary();
                left *= right;
            } else if (ch == '/') {
                self.pos += 1;
                const right = try self.parseUnary();
                if (right == 0) return error.DivisionByZero;
                left /= right;
            } else if (ch == '%') {
                self.pos += 1;
                const right = try self.parseUnary();
                if (right == 0) return error.DivisionByZero;
                left = @mod(left, right);
            } else break;
        }
        return left;
    }

    fn parseUnary(self: *ExprEval) anyerror!f64 {
        self.skipWs();
        if (self.pos < self.src.len and self.src[self.pos] == '-') {
            self.pos += 1;
            const v = try self.parseUnary();
            return -v;
        }
        if (self.pos < self.src.len and self.src[self.pos] == '+') {
            self.pos += 1;
            return self.parseUnary();
        }
        return self.parsePower();
    }

    fn parsePower(self: *ExprEval) anyerror!f64 {
        const base = try self.parseAtom();
        self.skipWs();
        if (self.pos + 1 < self.src.len and self.src[self.pos] == '*' and self.src[self.pos + 1] == '*') {
            self.pos += 2;
            const exp = try self.parseUnary();
            return std.math.pow(f64, base, exp);
        }
        if (self.pos < self.src.len and self.src[self.pos] == '^') {
            self.pos += 1;
            const exp = try self.parseUnary();
            return std.math.pow(f64, base, exp);
        }
        return base;
    }

    fn parseAtom(self: *ExprEval) anyerror!f64 {
        self.skipWs();
        if (self.pos >= self.src.len) return error.ParseFailure;
        const ch = self.src[self.pos];
        if (ch == '(') {
            self.pos += 1;
            const v = try self.parseExpr();
            self.skipWs();
            if (self.pos >= self.src.len or self.src[self.pos] != ')') return error.ParseFailure;
            self.pos += 1;
            return v;
        }
        if (std.ascii.isAlphabetic(ch)) {
            const start = self.pos;
            while (self.pos < self.src.len and std.ascii.isAlphabetic(self.src[self.pos])) self.pos += 1;
            const name = self.src[start..self.pos];
            self.skipWs();
            if (std.mem.eql(u8, name, "pi")) return std.math.pi;
            if (std.mem.eql(u8, name, "e")) return std.math.e;
            if (self.pos >= self.src.len or self.src[self.pos] != '(') return error.ParseFailure;
            self.pos += 1;
            const arg = try self.parseExpr();
            self.skipWs();
            if (self.pos >= self.src.len or self.src[self.pos] != ')') return error.ParseFailure;
            self.pos += 1;
            if (std.mem.eql(u8, name, "sqrt")) return @sqrt(arg);
            if (std.mem.eql(u8, name, "abs")) return @abs(arg);
            if (std.mem.eql(u8, name, "sin")) return @sin(arg);
            if (std.mem.eql(u8, name, "cos")) return @cos(arg);
            if (std.mem.eql(u8, name, "tan")) return @tan(arg);
            if (std.mem.eql(u8, name, "log")) {
                if (arg <= 0) return error.ParseFailure;
                return std.math.log(f64, std.math.e, arg);
            }
            if (std.mem.eql(u8, name, "exp")) return std.math.exp(arg);
            if (std.mem.eql(u8, name, "floor")) return @floor(arg);
            if (std.mem.eql(u8, name, "ceil")) return @ceil(arg);
            if (std.mem.eql(u8, name, "round")) return @round(arg);
            return error.ParseFailure;
        }
        const start = self.pos;
        var seen_dot = false;
        while (self.pos < self.src.len) {
            const cc = self.src[self.pos];
            if (std.ascii.isDigit(cc)) {
                self.pos += 1;
            } else if (cc == '.' and !seen_dot) {
                seen_dot = true;
                self.pos += 1;
            } else if ((cc == 'e' or cc == 'E') and self.pos > start) {
                if (self.pos + 1 < self.src.len and (std.ascii.isDigit(self.src[self.pos + 1]) or self.src[self.pos + 1] == '-' or self.src[self.pos + 1] == '+')) {
                    self.pos += 2;
                } else break;
            } else break;
        }
        if (self.pos == start) return error.ParseFailure;
        return std.fmt.parseFloat(f64, self.src[start..self.pos]) catch error.ParseFailure;
    }
};

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{ .thread_safe = true }){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    var config = try Config.load(allocator);
    defer config.deinit(allocator);
    try std.fs.cwd().makePath(config.workspace_root);
    try std.fs.cwd().makePath(config.knowledge_root);
    var db = try Database.open(allocator, config.database_path);
    defer db.close();
    try db.initSchema();
    try db.ensureTenant(config.default_tenant_id);
    var mgt = try MGT.init(allocator, SEED_VOCABULARY, SEED_ANCHORS, null, .dual);
    defer mgt.deinit();
    var ssi = SSI.init(allocator);
    defer ssi.deinit();
    var ranker = try Ranker.init(allocator, 4, 256, 0xA637200C4F12B551);
    defer ranker.deinit();
    var kernel = ChaosCoreKernel.init(allocator);
    defer kernel.deinit();
    var crev = try CREVPipeline.init(allocator, &kernel);
    defer crev.deinit();
    var app = App{
        .allocator = allocator,
        .config = config,
        .db = &db,
        .fs_mutex = .{},
        .retrieval_mutex = .{},
        .graph_mutex = .{},
        .mgt = mgt,
        .ssi = ssi,
        .ranker = ranker,
        .kernel = kernel,
        .crev = crev,
    };
    try seedInitialSkills(&app);
    try indexAllEnabledSkills(&app);
    try resumeActiveRuns(&app);
    const meta_thread = try std.Thread.spawn(.{}, metaHarnessEntry, .{&app});
    meta_thread.detach();
    try startHttpServer(&app);
}

const SEED_VOCABULARY: []const []const u8 = &.{
    "the",   "and",    "with",   "into",    "from",    "this",    "that",    "when",
    "where", "what",   "which",  "while",   "goal",    "task",    "state",   "step",
    "plan",  "planning", "execute", "executing", "verify", "verified", "finish", "done",
    "skill", "skills", "memory", "search",  "append",  "file",    "files",   "line",
    "lines", "unique", "log",    "logging", "write",   "read",    "check",   "compute",
    "result", "result", "number", "numeric", "arithmetic", "expression", "constraint", "constraints",
    "subgoal", "subgoals", "split", "record", "update", "status", "progress", "blocker",
    "blockers", "fact", "facts", "agent",  "runtime", "procedure", "trigger", "description",
    "work",  "working", "workspace", "knowledge", "query", "observation", "action", "error",
    "goal",  "decomposition", "verification", "hallucination", "duplicates", "duplicate", "cél", "feladat",
    "állapot", "lépés", "terv", "végrehajtás", "ellenőrzés", "kész", "képesség", "memória",
};

const SEED_ANCHORS: []const []const u8 = &.{
    "goal", "finish", "done", "verify", "plan",
};

fn seedInitialSkills(app: *App) !void {
    const tenant_id = app.config.default_tenant_id;
    try registerSkill(app, tenant_id, "skill_decompose_goal", "Goal Decomposition", "When status is planning or subgoals list is empty", "{\"trigger\":\"planning\"}", "{\"procedure\":\"1. Split goal into 3-7 subgoals. 2. Record constraints. 3. Update status to executing.\"}", true);
    try registerSkill(app, tenant_id, "skill_append_unique_log", "Append Unique Log Line", "When logging without duplicates", "{\"trigger\":\"log\"}", "{\"procedure\":\"1. Use filesystem.append_file with unique true. 2. Verify lines appended.\"}", true);
    try registerSkill(app, tenant_id, "skill_numeric_verification", "Numeric Verification", "When computing closed form arithmetic without hallucination", "{\"trigger\":\"arithmetic\"}", "{\"procedure\":\"1. Formulate mathematical expression. 2. Call compute tool. 3. Record verified result into facts.\"}", true);
}

fn skillNumericId(skill_id: []const u8) u64 {
    return stableHash(skill_id, SKILL_NUMERIC_SEED);
}

fn encodeSkillTokens(app: *App, name: []const u8, description: []const u8, trigger_json: []const u8, procedure_json: []const u8) ![]u32 {
    var combined = std.ArrayList(u8).init(app.allocator);
    defer combined.deinit();
    try combined.writer().print("{s}\n{s}\n{s}\n{s}", .{ name, description, trigger_json, procedure_json });
    var tokens = std.ArrayList(u32).init(app.allocator);
    errdefer tokens.deinit();
    app.retrieval_mutex.lock();
    defer app.retrieval_mutex.unlock();
    try app.mgt.encode(combined.items, &tokens);
    return tokens.toOwnedSlice();
}

fn tokensToJson(allocator: Allocator, tokens: []const u32) ![]u8 {
    var out = std.ArrayList(u8).init(allocator);
    errdefer out.deinit();
    const w = out.writer();
    try w.writeAll("[");
    for (tokens, 0..) |token, i| {
        if (i != 0) try w.writeAll(",");
        try w.print("{}", .{token});
    }
    try w.writeAll("]");
    return out.toOwnedSlice();
}

fn registerSkill(app: *App, tenant_id: []const u8, skill_id: []const u8, name: []const u8, description: []const u8, trigger_json: []const u8, procedure_json: []const u8, enabled: bool) !void {
    const tokens = try encodeSkillTokens(app, name, description, trigger_json, procedure_json);
    defer app.allocator.free(tokens);
    const tokens_json = try tokensToJson(app.allocator, tokens);
    defer app.allocator.free(tokens_json);
    try app.db.addSkillFromFields(tenant_id, skill_id, name, description, trigger_json, procedure_json, tokens_json, enabled);
    if (enabled) {
        app.retrieval_mutex.lock();
        defer app.retrieval_mutex.unlock();
        try app.ssi.addSequence(tokens, skillNumericId(skill_id), true);
    }
}

fn indexAllEnabledSkills(app: *App) !void {
    const skills = try app.db.loadEnabledSkills(app.config.default_tenant_id);
    defer freeSkillRecords(app.allocator, skills);
    for (skills) |skill| {
        const tokens = try encodeSkillTokens(app, skill.name, skill.description, skill.trigger_json, skill.procedure_json);
        defer app.allocator.free(tokens);
        app.retrieval_mutex.lock();
        try app.ssi.addSequence(tokens, skillNumericId(skill.id), true);
        app.retrieval_mutex.unlock();
    }
}

fn resumeActiveRuns(app: *App) !void {
    const ids = try app.db.listActiveRunIds();
    defer {
        for (ids) |id| app.allocator.free(id);
        app.allocator.free(ids);
    }
    for (ids) |id| try spawnRun(app, id);
}

fn spawnRun(app: *App, run_id: []const u8) !void {
    const r1 = try app.allocator.dupe(u8, run_id);
    const t2 = try std.Thread.spawn(.{}, system2Entry, .{ app, r1 });
    t2.detach();
    const r2 = try app.allocator.dupe(u8, run_id);
    const t1 = try std.Thread.spawn(.{}, system1Entry, .{ app, r2 });
    t1.detach();
}

fn system2Entry(app: *App, run_id: []u8) void {
    defer app.allocator.free(run_id);
    runSystem2(app, run_id) catch |err| {
        var err_json: []const u8 = "{\"error\":\"system2\"}";
        var err_json_owned = false;
        if (std.fmt.allocPrint(app.allocator, "{{\"error\":\"{s}\"}}", .{@errorName(err)})) |allocated| {
            err_json = allocated;
            err_json_owned = true;
        } else |_| {}
        app.db.markRunStatus(run_id, "failed", err_json) catch {};
        if (err_json_owned) app.allocator.free(err_json);
    };
    finalizeTrajectory(app, run_id) catch {};
}

fn system1Entry(app: *App, run_id: []u8) void {
    defer app.allocator.free(run_id);
    runSystem1(app, run_id) catch |err| {
        const err_json = std.fmt.allocPrint(app.allocator, "{{\"error\":\"{s}\"}}", .{@errorName(err)}) catch return;
        defer app.allocator.free(err_json);
        _ = app.db.insertEvent(app.config.default_tenant_id, run_id, 0, "system1_error", err_json) catch {};
    };
}

fn runSystem2(app: *App, run_id: []const u8) !void {
    while (true) {
        var run = app.db.getRun(run_id) catch |err| {
            if (err == error.NotFound) return;
            return err;
        };
        defer run.deinit(app.allocator);
        if (!std.mem.eql(u8, run.status, "running") and !std.mem.eql(u8, run.status, "queued")) break;
        if (run.step >= app.config.max_steps) {
            const err_json = try std.fmt.allocPrint(app.allocator, "{{\"error\":\"max_steps_exceeded\",\"max_steps\":{}}}", .{app.config.max_steps});
            defer app.allocator.free(err_json);
            try app.db.markRunStatus(run_id, "failed", err_json);
            break;
        }
        if (try app.db.hasOpenAction(run_id)) {
            std.time.sleep(50 * std.time.ns_per_ms);
            continue;
        }
        const initial_state = try app.db.getStateJson(run_id);
        defer app.allocator.free(initial_state);
        if (initial_state.len > app.config.max_state_bytes) return error.StateTooLarge;
        const observation = try app.db.takeObservation(run_id);
        defer app.allocator.free(observation);
        const skills = try routeSkills(app, run.tenant_id, initial_state, observation);
        defer freeSkillRecords(app.allocator, skills);
        const skills_json = try selectedSkillsJson(app.allocator, skills);
        defer app.allocator.free(skills_json);
        const prompt = try buildStepPrompt(app.allocator, run.procedure_json, initial_state, observation, skills_json);
        defer app.allocator.free(prompt);
        if (prompt.len > app.config.max_prompt_bytes) return error.PromptTooLarge;

        const frame = try buildCognitionFrame(app, initial_state, observation, run.step);
        defer app.allocator.free(frame);
        try app.db.insertCognitionFrame(run_id, run.step, frame);

        var model_result = try callModelStreaming(app, prompt);
        defer model_result.deinit(app.allocator);
        const token_charge = if (model_result.total_tokens > 0) model_result.total_tokens else estimateTokens(prompt.len + model_result.content.len);
        try app.db.incrementTokenUsage(run_id, token_charge);

        var envelope = validateEnvelope(app, model_result.content) catch |err| {
            var full = std.ArrayList(u8).init(app.allocator);
            defer full.deinit();
            const w = full.writer();
            try w.print("{{\"error\":\"{s}\",\"model_output\":", .{@errorName(err)});
            try writeJsonString(w, model_result.content);
            try w.writeAll("}");
            const err_payload = try full.toOwnedSlice();
            defer app.allocator.free(err_payload);
            _ = try app.db.insertEvent(run.tenant_id, run_id, run.step, "validation_failed", err_payload);
            const obs = try std.fmt.allocPrint(app.allocator, "{{\"type\":\"validation_error\",\"error\":\"{s}\",\"instruction\":\"Return valid JSON envelope with state_patch and action.\"}}", .{@errorName(err)});
            defer app.allocator.free(obs);
            try app.db.updateLatestObservation(run_id, obs);
            std.time.sleep(500 * std.time.ns_per_ms);
            continue;
        };
        defer envelope.deinit(app.allocator);

        const next_step = run.step + 1;
        const post_state = try app.db.applyPatchAndCheckpoint(run_id, next_step, envelope.patch_json, observation, envelope.action_json, app.config.max_state_bytes);
        defer app.allocator.free(post_state);

        const action_id = try app.db.enqueueAction(run_id, next_step, envelope.action_json);
        const outcome = try waitActionOutcome(app, action_id, 600000);
        defer app.allocator.free(outcome);

        const post_action_state = try app.db.getStateJson(run_id);
        defer app.allocator.free(post_action_state);
        const verifier = try evaluateStepVerifier(app.allocator, post_action_state, outcome);
        defer app.allocator.free(verifier);

        try app.db.addRawTrace(run.tenant_id, run_id, next_step, initial_state, skills_json, observation, envelope.envelope_json, envelope.patch_json, envelope.action_json, outcome, post_action_state, verifier);
        _ = try app.db.insertEvent(run.tenant_id, run_id, next_step, "step_completed", outcome);

        if (envelope.terminal) {
            const terminal_result = try std.fmt.allocPrint(app.allocator, "{{\"terminal\":true,\"confidence\":{d},\"step\":{},\"action\":{s}}}", .{ envelope.confidence, next_step, envelope.action_json });
            defer app.allocator.free(terminal_result);
            try app.db.markCompleted(run_id, terminal_result);
            _ = try app.db.insertEvent(run.tenant_id, run_id, next_step, "terminal_envelope", terminal_result);
            break;
        }

        const status = try app.db.getRunStatus(run_id);
        defer app.allocator.free(status);
        if (!std.mem.eql(u8, status, "running") and !std.mem.eql(u8, status, "queued")) break;
        std.time.sleep(50 * std.time.ns_per_ms);
    }
}

fn runSystem1(app: *App, run_id: []const u8) !void {
    while (true) {
        const status = try app.db.getRunStatus(run_id);
        defer app.allocator.free(status);
        if (!std.mem.eql(u8, status, "running") and !std.mem.eql(u8, status, "queued")) break;
        if (try app.db.claimPendingAction(run_id)) |claim| {
            var c = claim;
            defer c.deinit(app.allocator);
            const outcome = executeAuthorizedAction(app, c.run_id, c.step, c.action_json) catch |err| blk: {
                break :blk try std.fmt.allocPrint(app.allocator, "{{\"ok\":false,\"type\":\"tool_error\",\"error\":\"{s}\",\"at\":{}}}", .{ @errorName(err), now() });
            };
            defer app.allocator.free(outcome);
            try app.db.completeAction(c.id, outcome);
            try app.db.updateLatestObservation(c.run_id, outcome);
        }
        std.time.sleep(20 * std.time.ns_per_ms);
    }
}

fn waitActionOutcome(app: *App, action_id: i64, timeout_ms: i64) ![]u8 {
    const start = nowMillis();
    while (nowMillis() - start < timeout_ms) {
        if (try app.db.getActionResult(action_id)) |result| return result;
        std.time.sleep(20 * std.time.ns_per_ms);
    }
    return error.ActionTimeout;
}

fn routeSkills(app: *App, tenant_id: []const u8, state_json: []const u8, observation_json: []const u8) ![]SkillRecord {
    const allocator = app.allocator;
    var query_builder = std.ArrayList(u8).init(allocator);
    defer query_builder.deinit();
    try query_builder.writer().print("{s}\n{s}", .{ state_json, observation_json });

    var query_tokens = std.ArrayList(u32).init(allocator);
    defer query_tokens.deinit();

    var candidates: []RankedSegment = try allocator.alloc(RankedSegment, 0);
    var candidates_owned = true;
    defer {
        if (candidates_owned) {
            freeRankedSegments(candidates, allocator);
        }
    }

    var fts_ids: [][]u8 = try allocator.alloc([]u8, 0);
    var fts_owned = true;
    defer {
        if (fts_owned) {
            for (fts_ids) |id| allocator.free(id);
            allocator.free(fts_ids);
        }
    }

    app.retrieval_mutex.lock();
    const encode_err = app.mgt.encode(query_builder.items, &query_tokens);
    app.retrieval_mutex.unlock();
    try encode_err;

    if (query_tokens.items.len > 0) {
        app.retrieval_mutex.lock();
        const retrieve_err = app.ssi.retrieveTopK(query_tokens.items, ROUTE_TOP_K, allocator);
        app.retrieval_mutex.unlock();
        const retrieved = try retrieve_err;
        freeRankedSegments(candidates, allocator);
        candidates = retrieved;
    }

    if (candidates.len > 0 and query_tokens.items.len > 0) {
        app.retrieval_mutex.lock();
        const rerank_err = app.ranker.rankCandidatesWithQuery(candidates, query_tokens.items, &app.ssi, allocator);
        app.retrieval_mutex.unlock();
        try rerank_err;
    }

    const fts_query = try makeFtsQuery(allocator, query_builder.items);
    defer allocator.free(fts_query);
    if (fts_query.len > 0) {
        const searched = app.db.searchSkillFtsIds(tenant_id, fts_query) catch blk: {
            break :blk try allocator.alloc([]u8, 0);
        };
        for (fts_ids) |id| allocator.free(id);
        allocator.free(fts_ids);
        fts_ids = searched;
    }

    const skills = try app.db.loadEnabledSkills(tenant_id);
    var skills_owned = true;
    errdefer {
        if (skills_owned) {
            freeSkillRecords(allocator, skills);
        }
    }

    for (skills) |*skill| {
        const numeric_id = skillNumericId(skill.id);
        var ranker_rank: usize = std.math.maxInt(usize);
        var ranker_score: f64 = 0.0;
        for (candidates, 0..) |candidate, candidate_index| {
            if (candidate.position == numeric_id) {
                ranker_rank = candidate_index;
                ranker_score = candidate.score;
                break;
            }
        }
        skill.vector_score = ranker_score;
        skill.sparse_rank = findRank(fts_ids, skill.id);
        const ranker_component = if (ranker_rank == std.math.maxInt(usize)) 0.0 else 1.0 / @as(f64, @floatFromInt(60 + ranker_rank + 1));
        const sparse_component = if (skill.sparse_rank == std.math.maxInt(usize)) 0.0 else 1.0 / @as(f64, @floatFromInt(60 + skill.sparse_rank + 1));
        skill.rrf_score = sparse_component + ranker_component + ranker_score * 0.01;
    }

    var selected = std.ArrayList(SkillRecord).init(allocator);
    errdefer {
        for (selected.items) |*record| record.deinit(allocator);
        selected.deinit();
    }
    var taken = try allocator.alloc(bool, skills.len);
    defer allocator.free(taken);
    @memset(taken, false);
    var round: usize = 0;
    while (round < 2 and round < skills.len) : (round += 1) {
        var best_index: ?usize = null;
        var best_score: f64 = -1000000.0;
        for (skills, 0..) |skill, i| {
            if (taken[i]) continue;
            if (skill.rrf_score > best_score) {
                best_score = skill.rrf_score;
                best_index = i;
            }
        }
        if (best_index) |bi| {
            taken[bi] = true;
            const s = skills[bi];
            try selected.append(SkillRecord{
                .id = try allocator.dupe(u8, s.id),
                .name = try allocator.dupe(u8, s.name),
                .description = try allocator.dupe(u8, s.description),
                .trigger_json = try allocator.dupe(u8, s.trigger_json),
                .procedure_json = try allocator.dupe(u8, s.procedure_json),
                .embedding_json = try allocator.dupe(u8, s.embedding_json),
                .vector_score = s.vector_score,
                .sparse_rank = s.sparse_rank,
                .rrf_score = best_score,
            });
        }
    }

    freeSkillRecords(allocator, skills);
    skills_owned = false;
    freeRankedSegments(candidates, allocator);
    candidates_owned = false;
    for (fts_ids) |id| allocator.free(id);
    allocator.free(fts_ids);
    fts_owned = false;
    return selected.toOwnedSlice();
}

fn findRank(ids: [][]u8, id: []const u8) usize {
    for (ids, 0..) |candidate, i| if (std.mem.eql(u8, candidate, id)) return i;
    return std.math.maxInt(usize);
}

fn freeSkillRecords(allocator: Allocator, records: []SkillRecord) void {
    for (records) |*record| record.deinit(allocator);
    allocator.free(records);
}

fn selectedSkillsJson(allocator: Allocator, skills: []SkillRecord) ![]u8 {
    var out = std.ArrayList(u8).init(allocator);
    const w = out.writer();
    try w.writeAll("[");
    for (skills, 0..) |skill, i| {
        if (i != 0) try w.writeAll(",");
        try w.writeAll("{\"id\":");
        try writeJsonString(w, skill.id);
        try w.writeAll(",\"name\":");
        try writeJsonString(w, skill.name);
        try w.writeAll(",\"description\":");
        try writeJsonString(w, skill.description);
        try w.writeAll(",\"trigger\":");
        try w.writeAll(skill.trigger_json);
        try w.writeAll(",\"procedure\":");
        try w.writeAll(skill.procedure_json);
        try w.print(",\"retrieval_score\":{}}}", .{skill.rrf_score});
    }
    try w.writeAll("]");
    return out.toOwnedSlice();
}

fn buildStepPrompt(allocator: Allocator, procedure_json: []const u8, state_json: []const u8, observation_json: []const u8, skills_json: []const u8) ![]u8 {
    var out = std.ArrayList(u8).init(allocator);
    const w = out.writer();
    try w.writeAll("{\"runtime\":\"autonomous_state_agent\",\"contract\":{\"prompt_footprint\":\"O(1)\",\"history_policy\":\"No append-only conversation history or reasoning logs permitted.\",\"state_patch_semantics\":\"Top-level merge. Null values delete keys.\",\"output\":\"Return exactly one JSON object with keys state_patch, action, terminal, confidence.\"},\"procedure\":");
    try w.writeAll(procedure_json);
    try w.writeAll(",\"working_memory_state\":");
    try w.writeAll(state_json);
    try w.writeAll(",\"latest_environment_observation\":");
    try w.writeAll(observation_json);
    try w.writeAll(",\"experiential_memory_skills\":");
    try w.writeAll(skills_json);
    try w.writeAll(",\"allowed_actions\":[\"none\",\"finish\",\"sleep\",\"emit\",\"filesystem.write_file\",\"filesystem.read_file\",\"filesystem.append_file\",\"filesystem.replace_lines\",\"filesystem.check_lines\",\"filesystem.list_dir\",\"filesystem.delete_file\",\"compute\",\"memory.search\",\"memory.add_skill\",\"knowledge.query\",\"http.get\"],\"required_schema\":{\"state_patch\":\"object\",\"action\":{\"type\":\"string\",\"args\":\"object\"},\"terminal\":\"boolean\",\"confidence\":\"number\"}}");
    return out.toOwnedSlice();
}

fn buildCognitionFrame(app: *App, state_json: []const u8, observation_json: []const u8, step: i64) ![]u8 {
    const allocator = app.allocator;
    var combined = std.ArrayList(u8).init(allocator);
    defer combined.deinit();
    try combined.writer().print("{s}\n{s}", .{ state_json, observation_json });
    var tokens = std.ArrayList(u32).init(allocator);
    defer tokens.deinit();
    app.retrieval_mutex.lock();
    const encode_err = app.mgt.encode(combined.items, &tokens);
    app.retrieval_mutex.unlock();
    try encode_err;
    const signature = SSI.computeMinHashSignature(tokens.items);
    var token_bytes: [4]u8 = undefined;
    var out = std.ArrayList(u8).init(allocator);
    errdefer out.deinit();
    const w = out.writer();
    try w.print("{{\"step\":{},\"generated_at\":{},\"matrix\":[", .{ step, nowMillis() });
    var k: usize = 0;
    while (k < COG_K) : (k += 1) {
        if (k != 0) try w.writeAll(",");
        try w.writeAll("[");
        var h: usize = 0;
        while (h < COG_H) : (h += 1) {
            if (h != 0) try w.writeAll(",");
            const cell_index = k * COG_H + h;
            std.mem.writeInt(u32, &token_bytes, @as(u32, @truncate(cell_index)), .little);
            const cell_hash = stableHash(&token_bytes, signature +% @as(u64, cell_index));
            const cell_value = (@as(f64, @floatFromInt(cell_hash & 0x1FFFFFFFFFFFFF)) / @as(f64, @floatFromInt(@as(u64, 1) << 53))) * 2.0 - 1.0;
            try w.print("{d}", .{cell_value});
        }
        try w.writeAll("]");
    }
    const gate = if (containsIgnoreCase(state_json, "\"done\":true")) "verify" else "continue";
    try w.writeAll("],\"transition_gate\":");
    try writeJsonString(w, gate);
    try w.writeAll("}");
    return out.toOwnedSlice();
}

fn callModelStreaming(app: *App, user_prompt: []const u8) !ModelResult {
    const system_prompt = "You are an autonomous runtime policy. Reason privately, discard intermediate chain of thought, and output valid plain JSON matching the schema.";
    var client = std.http.Client{ .allocator = app.allocator };
    defer client.deinit();

    const url = try joinUrl(app.allocator, app.config.modular_base_url, "chat/completions");
    defer app.allocator.free(url);
    const uri = try std.Uri.parse(url);

    var body_buf = std.ArrayList(u8).init(app.allocator);
    defer body_buf.deinit();
    const bw = body_buf.writer();
    try bw.writeAll("{\"model\":");
    try writeJsonString(bw, app.config.model);
    try bw.writeAll(",\"messages\":[{\"role\":\"system\",\"content\":");
    try writeJsonString(bw, system_prompt);
    try bw.writeAll("},{\"role\":\"user\",\"content\":");
    try writeJsonString(bw, user_prompt);
    try bw.print("]}},\"stream\":true,\"stream_options\":{{\"include_usage\":true}},\"temperature\":0.7,\"max_tokens\":{},\"response_format\":{{\"type\":\"json_object\"}}}}", .{app.config.model_max_tokens});

    const auth = try std.fmt.allocPrint(app.allocator, "Bearer {s}", .{app.config.modular_api_key});
    defer app.allocator.free(auth);

    var header_buf: [8192]u8 = undefined;
    var req = try client.open(.POST, uri, .{
        .server_header_buffer = &header_buf,
        .extra_headers = &.{
            .{ .name = "authorization", .value = auth },
            .{ .name = "content-type", .value = "application/json" },
            .{ .name = "accept", .value = "text/event-stream" },
        },
    });
    defer req.deinit();

    req.transfer_encoding = .{ .content_length = body_buf.items.len };
    try req.send();
    try req.writeAll(body_buf.items);
    try req.finish();
    try req.wait();

    if (req.response.status != .ok) return error.ModelRequestFailed;
    return parseModelResponseStream(app.allocator, req.reader());
}

fn consumeBufferedLine(line_buf: *std.ArrayList(u8), newline_index: usize) void {
    const remaining = line_buf.items.len - (newline_index + 1);
    std.mem.copyForwards(u8, line_buf.items[0..remaining], line_buf.items[newline_index + 1 ..]);
    line_buf.shrinkRetainingCapacity(remaining);
}

fn parseModelResponseStream(allocator: Allocator, reader: anytype) !ModelResult {
    var content = std.ArrayList(u8).init(allocator);
    errdefer content.deinit();
    var total_tokens: i64 = 0;
    var line_buf = std.ArrayList(u8).init(allocator);
    defer line_buf.deinit();
    var chunk: [4096]u8 = undefined;

    while (true) {
        const n = reader.read(&chunk) catch break;
        if (n == 0) break;
        try line_buf.appendSlice(chunk[0..n]);
        while (std.mem.indexOfScalar(u8, line_buf.items, '\n')) |nl| {
            const raw_line = line_buf.items[0..nl];
            const line = std.mem.trim(u8, raw_line, " \t\r");
            if (std.mem.startsWith(u8, line, "data:")) {
                const payload = std.mem.trim(u8, line[5..], " \t");
                if (!std.mem.eql(u8, payload, "[DONE]")) {
                    var parsed = std.json.parseFromSlice(std.json.Value, allocator, payload, .{}) catch {
                        consumeBufferedLine(&line_buf, nl);
                        continue;
                    };
                    defer parsed.deinit();
                    if (objectGetValue(parsed.value, "usage")) |usage| {
                        if (objectGetInt(usage, "total_tokens")) |t| total_tokens = t;
                    }
                    if (objectGetValue(parsed.value, "choices")) |choices| {
                        switch (choices) {
                            .array => |arr| {
                                if (arr.items.len > 0) {
                                    if (objectGetValue(arr.items[0], "delta")) |delta| {
                                        if (objectGetString(delta, "content")) |s| try content.appendSlice(s);
                                    }
                                }
                            },
                            else => {},
                        }
                    }
                }
            }
            consumeBufferedLine(&line_buf, nl);
        }
    }
    return ModelResult{ .content = try content.toOwnedSlice(), .total_tokens = total_tokens };
}

fn validateEnvelope(app: *App, model_content: []const u8) !Envelope {
    const extracted = try extractJsonObject(app.allocator, model_content);
    defer app.allocator.free(extracted);
    var parsed = try std.json.parseFromSlice(std.json.Value, app.allocator, extracted, .{});
    defer parsed.deinit();
    try validateValueBounded(parsed.value, 0);
    const patch_value = objectGetValue(parsed.value, "state_patch") orelse return error.MissingStatePatch;
    const action_value = objectGetValue(parsed.value, "action") orelse return error.MissingAction;
    try validatePatchValue(patch_value);
    try validateActionValue(action_value);
    const patch_json = try jsonValueToOwned(app.allocator, patch_value);
    const action_json = try jsonValueToOwned(app.allocator, action_value);
    const envelope_json = try jsonValueToOwned(app.allocator, parsed.value);
    return Envelope{
        .envelope_json = envelope_json,
        .patch_json = patch_json,
        .action_json = action_json,
        .terminal = objectGetBool(parsed.value, "terminal") orelse false,
        .confidence = objectGetFloat(parsed.value, "confidence") orelse 0.0,
    };
}

fn extractJsonObject(allocator: Allocator, text: []const u8) ![]u8 {
    const trimmed = std.mem.trim(u8, text, " \t\r\n");
    if (trimmed.len >= 2 and trimmed[0] == '{' and trimmed[trimmed.len - 1] == '}') return allocator.dupe(u8, trimmed);
    var start: ?usize = null;
    var depth: i64 = 0;
    var in_string = false;
    var escaped = false;
    var i: usize = 0;
    while (i < text.len) : (i += 1) {
        const ch = text[i];
        if (in_string) {
            if (escaped) {
                escaped = false;
            } else if (ch == '\\') {
                escaped = true;
            } else if (ch == '"') {
                in_string = false;
            }
            continue;
        }
        if (ch == '"') {
            in_string = true;
        } else if (ch == '{') {
            if (depth == 0) start = i;
            depth += 1;
        } else if (ch == '}') {
            depth -= 1;
            if (depth == 0 and start != null) return allocator.dupe(u8, text[start.? .. i + 1]);
        }
    }
    return error.NoJsonObject;
}

fn validateActionValue(value: std.json.Value) !void {
    const typ = objectGetString(value, "type") orelse return error.ActionMissingType;
    if (std.mem.eql(u8, typ, "none") or std.mem.eql(u8, typ, "finish") or std.mem.eql(u8, typ, "sleep") or std.mem.eql(u8, typ, "emit") or std.mem.eql(u8, typ, "filesystem.write_file") or std.mem.eql(u8, typ, "filesystem.read_file") or std.mem.eql(u8, typ, "filesystem.append_file") or std.mem.eql(u8, typ, "filesystem.replace_lines") or std.mem.eql(u8, typ, "filesystem.check_lines") or std.mem.eql(u8, typ, "filesystem.list_dir") or std.mem.eql(u8, typ, "filesystem.delete_file") or std.mem.eql(u8, typ, "compute") or std.mem.eql(u8, typ, "memory.search") or std.mem.eql(u8, typ, "memory.add_skill") or std.mem.eql(u8, typ, "knowledge.query") or std.mem.eql(u8, typ, "http.get")) {
        try validateValueBounded(value, 0);
        return;
    }
    return error.ActionNotAllowed;
}

fn executeAuthorizedAction(app: *App, run_id: []const u8, step: i64, action_json: []const u8) ![]u8 {
    var parsed = try std.json.parseFromSlice(std.json.Value, app.allocator, action_json, .{});
    defer parsed.deinit();
    const typ = objectGetString(parsed.value, "type") orelse return error.ActionMissingType;
    const args = objectGetValue(parsed.value, "args") orelse parsed.value;
    const generated_at = try app.db.latestCognitionGeneratedAt(run_id);
    const elapsed_ms = nowMillis() - generated_at;

    var staleness_buf = std.ArrayList(u8).init(app.allocator);
    defer staleness_buf.deinit();
    const sw = staleness_buf.writer();
    try sw.writeAll("[");
    var k: usize = 0;
    while (k < 4) : (k += 1) {
        const freq = std.math.pow(f64, 10000.0, @as(f64, @floatFromInt(k)) / 4.0);
        const t = @as(f64, @floatFromInt(elapsed_ms));
        if (k != 0) try sw.writeAll(",");
        try sw.print("{{\"sin\":{d:.4},\"cos\":{d:.4}}}", .{ @sin(t / freq), @cos(t / freq) });
    }
    try sw.writeAll("]");

    if (std.mem.eql(u8, typ, "none")) {
        return std.fmt.allocPrint(app.allocator, "{{\"ok\":true,\"type\":\"none\",\"step\":{},\"staleness\":{s},\"at\":{}}}", .{ step, staleness_buf.items, now() });
    }
    if (std.mem.eql(u8, typ, "finish")) {
        const result_value = objectGetValue(args, "result") orelse args;
        const result_json = try jsonValueToOwned(app.allocator, result_value);
        defer app.allocator.free(result_json);
        try app.db.markCompleted(run_id, result_json);
        return std.fmt.allocPrint(app.allocator, "{{\"ok\":true,\"type\":\"finish\",\"result\":{s},\"step\":{},\"staleness\":{s},\"at\":{}}}", .{ result_json, step, staleness_buf.items, now() });
    }
    if (std.mem.eql(u8, typ, "sleep")) {
        const ms = objectGetInt(args, "ms") orelse 1000;
        const clamped_ms = @min(@max(ms, 0), 60000);
        std.time.sleep(@as(u64, @intCast(clamped_ms)) * std.time.ns_per_ms);
        return std.fmt.allocPrint(app.allocator, "{{\"ok\":true,\"type\":\"sleep\",\"ms\":{},\"step\":{},\"staleness\":{s},\"at\":{}}}", .{ clamped_ms, step, staleness_buf.items, now() });
    }
    if (std.mem.eql(u8, typ, "emit")) {
        const payload_value = objectGetValue(args, "payload") orelse args;
        const payload_json = try jsonValueToOwned(app.allocator, payload_value);
        defer app.allocator.free(payload_json);
        var run = try app.db.getRun(run_id);
        defer run.deinit(app.allocator);
        _ = try app.db.insertEvent(run.tenant_id, run_id, step, "agent_emit", payload_json);
        routeTextThroughKnowledge(app, run_id, payload_json);
        return std.fmt.allocPrint(app.allocator, "{{\"ok\":true,\"type\":\"emit\",\"payload\":{s},\"step\":{},\"staleness\":{s},\"at\":{}}}", .{ payload_json, step, staleness_buf.items, now() });
    }
    if (std.mem.eql(u8, typ, "compute")) {
        const expr = objectGetString(args, "expression") orelse return error.MissingExpression;
        var ev = ExprEval{ .src = expr, .pos = 0 };
        const val = try ev.parseExpr();
        ev.skipWs();
        if (ev.pos != expr.len) return error.TrailingTokens;
        var out = std.ArrayList(u8).init(app.allocator);
        errdefer out.deinit();
        const w = out.writer();
        try w.writeAll("{\"ok\":true,\"type\":\"compute\",\"expression\":");
        try writeJsonString(w, expr);
        try w.print(",\"result\":{d},\"step\":{d},\"staleness\":{s},\"at\":{d}}}", .{ val, step, staleness_buf.items, now() });
        return out.toOwnedSlice();
    }
    if (std.mem.eql(u8, typ, "memory.search")) {
        const query = objectGetString(args, "query") orelse return error.MissingQuery;
        var run = try app.db.getRun(run_id);
        defer run.deinit(app.allocator);
        const skills = try routeSkills(app, run.tenant_id, query, "{}");
        defer freeSkillRecords(app.allocator, skills);
        const sj = try selectedSkillsJson(app.allocator, skills);
        defer app.allocator.free(sj);
        return std.fmt.allocPrint(app.allocator, "{{\"ok\":true,\"type\":\"memory.search\",\"skills\":{s},\"step\":{},\"staleness\":{s},\"at\":{}}}", .{ sj, step, staleness_buf.items, now() });
    }
    if (std.mem.eql(u8, typ, "memory.add_skill")) {
        var run = try app.db.getRun(run_id);
        defer run.deinit(app.allocator);
        const skill_id = try makeId(app.allocator, "skill");
        defer app.allocator.free(skill_id);
        const name = objectGetString(args, "name") orelse return error.SkillMissingName;
        const description = objectGetString(args, "description") orelse return error.SkillMissingDescription;
        const trigger_value = objectGetValue(args, "trigger") orelse std.json.Value{ .object = std.json.ObjectMap.init(app.allocator) };
        const procedure_value = objectGetValue(args, "procedure") orelse return error.SkillMissingProcedure;
        const trigger_json = try jsonValueToOwned(app.allocator, trigger_value);
        defer app.allocator.free(trigger_json);
        const procedure_json = try jsonValueToOwned(app.allocator, procedure_value);
        defer app.allocator.free(procedure_json);
        try registerSkill(app, run.tenant_id, skill_id, name, description, trigger_json, procedure_json, true);
        var out = std.ArrayList(u8).init(app.allocator);
        errdefer out.deinit();
        const w = out.writer();
        try w.writeAll("{\"ok\":true,\"type\":\"memory.add_skill\",\"skill_id\":");
        try writeJsonString(w, skill_id);
        try w.print(",\"step\":{d},\"staleness\":{s},\"at\":{d}}}", .{ step, staleness_buf.items, now() });
        return out.toOwnedSlice();
    }
    if (std.mem.eql(u8, typ, "filesystem.write_file")) {
        const path = objectGetString(args, "path") orelse return error.MissingPath;
        const content = objectGetString(args, "content") orelse return error.MissingContent;
        app.fs_mutex.lock();
        defer app.fs_mutex.unlock();
        const resolved = try resolveWorkspacePath(app.allocator, app.config.workspace_root, path);
        defer app.allocator.free(resolved);
        try atomicWriteFile(app.allocator, resolved, content);
        routeTextThroughKnowledge(app, run_id, content);
        var out = std.ArrayList(u8).init(app.allocator);
        errdefer out.deinit();
        const w = out.writer();
        try w.writeAll("{\"ok\":true,\"type\":\"filesystem.write_file\",\"path\":");
        try writeJsonString(w, path);
        try w.print(",\"bytes_written\":{d},\"step\":{d},\"staleness\":{s},\"at\":{d}}}", .{ content.len, step, staleness_buf.items, now() });
        return out.toOwnedSlice();
    }
    if (std.mem.eql(u8, typ, "filesystem.read_file")) {
        const path = objectGetString(args, "path") orelse return error.MissingPath;
        app.fs_mutex.lock();
        defer app.fs_mutex.unlock();
        const resolved = try resolveWorkspacePath(app.allocator, app.config.workspace_root, path);
        defer app.allocator.free(resolved);
        const data = try std.fs.cwd().readFileAlloc(app.allocator, resolved, 16 * 1024 * 1024);
        defer app.allocator.free(data);
        var out = std.ArrayList(u8).init(app.allocator);
        errdefer out.deinit();
        const w = out.writer();
        try w.writeAll("{\"ok\":true,\"type\":\"filesystem.read_file\",\"path\":");
        try writeJsonString(w, path);
        try w.writeAll(",\"content\":");
        try writeJsonString(w, data);
        try w.print(",\"step\":{d},\"staleness\":{s},\"at\":{d}}}", .{ step, staleness_buf.items, now() });
        return out.toOwnedSlice();
    }
    if (std.mem.eql(u8, typ, "filesystem.append_file")) {
        const path = objectGetString(args, "path") orelse return error.MissingPath;
        const content = objectGetString(args, "content") orelse return error.MissingContent;
        const unique = objectGetBool(args, "unique") orelse false;
        app.fs_mutex.lock();
        defer app.fs_mutex.unlock();
        const resolved = try resolveWorkspacePath(app.allocator, app.config.workspace_root, path);
        defer app.allocator.free(resolved);
        const appended = try appendFileLineOriented(app.allocator, resolved, content, unique);
        routeTextThroughKnowledge(app, run_id, content);
        var out = std.ArrayList(u8).init(app.allocator);
        errdefer out.deinit();
        const w = out.writer();
        try w.writeAll("{\"ok\":true,\"type\":\"filesystem.append_file\",\"path\":");
        try writeJsonString(w, path);
        try w.print(",\"lines_appended\":{d},\"step\":{d},\"staleness\":{s},\"at\":{d}}}", .{ appended, step, staleness_buf.items, now() });
        return out.toOwnedSlice();
    }
    if (std.mem.eql(u8, typ, "filesystem.replace_lines")) {
        const path = objectGetString(args, "path") orelse return error.MissingPath;
        const start_line = objectGetInt(args, "start_line") orelse return error.MissingLineRange;
        const end_line = objectGetInt(args, "end_line") orelse return error.MissingLineRange;
        const content = objectGetString(args, "content") orelse return error.MissingContent;
        if (start_line < 1 or end_line < start_line) return error.InvalidLineRange;
        app.fs_mutex.lock();
        defer app.fs_mutex.unlock();
        const resolved = try resolveWorkspacePath(app.allocator, app.config.workspace_root, path);
        defer app.allocator.free(resolved);
        try replaceLines(app.allocator, resolved, @intCast(start_line), @intCast(end_line), content);
        var out = std.ArrayList(u8).init(app.allocator);
        errdefer out.deinit();
        const w = out.writer();
        try w.writeAll("{\"ok\":true,\"type\":\"filesystem.replace_lines\",\"path\":");
        try writeJsonString(w, path);
        try w.print(",\"start_line\":{d},\"end_line\":{d},\"step\":{d},\"staleness\":{s},\"at\":{d}}}", .{ start_line, end_line, step, staleness_buf.items, now() });
        return out.toOwnedSlice();
    }
    if (std.mem.eql(u8, typ, "filesystem.check_lines")) {
        const path = objectGetString(args, "path") orelse return error.MissingPath;
        const lines_value = objectGetValue(args, "lines") orelse return error.MissingLines;
        app.fs_mutex.lock();
        defer app.fs_mutex.unlock();
        const resolved = try resolveWorkspacePath(app.allocator, app.config.workspace_root, path);
        defer app.allocator.free(resolved);
        const result = try checkLines(app.allocator, resolved, lines_value);
        defer app.allocator.free(result);
        var out = std.ArrayList(u8).init(app.allocator);
        errdefer out.deinit();
        const w = out.writer();
        try w.writeAll("{\"ok\":true,\"type\":\"filesystem.check_lines\",\"path\":");
        try writeJsonString(w, path);
        try w.print(",\"result\":{s},\"step\":{d},\"staleness\":{s},\"at\":{d}}}", .{ result, step, staleness_buf.items, now() });
        return out.toOwnedSlice();
    }
    if (std.mem.eql(u8, typ, "filesystem.list_dir")) {
        var path = objectGetString(args, "path") orelse ".";
        if (path.len == 0) path = ".";
        app.fs_mutex.lock();
        defer app.fs_mutex.unlock();
        const resolved = try resolveWorkspacePath(app.allocator, app.config.workspace_root, path);
        defer app.allocator.free(resolved);
        var dir = try std.fs.cwd().openDir(resolved, .{ .iterate = true });
        defer dir.close();
        var list = std.ArrayList(u8).init(app.allocator);
        defer list.deinit();
        const lw = list.writer();
        try lw.writeAll("[");
        var first = true;
        var it = dir.iterate();
        while (try it.next()) |entry| {
            if (!first) try lw.writeAll(",");
            first = false;
            try lw.writeAll("{\"name\":");
            try writeJsonString(lw, entry.name);
            try lw.writeAll(",\"is_dir\":");
            try lw.writeAll(if (entry.kind == .directory) "true" else "false");
            try lw.writeAll("}");
        }
        try lw.writeAll("]");
        var out = std.ArrayList(u8).init(app.allocator);
        errdefer out.deinit();
        const w = out.writer();
        try w.writeAll("{\"ok\":true,\"type\":\"filesystem.list_dir\",\"path\":");
        try writeJsonString(w, path);
        try w.print(",\"entries\":{s},\"step\":{d},\"staleness\":{s},\"at\":{d}}}", .{ list.items, step, staleness_buf.items, now() });
        return out.toOwnedSlice();
    }
    if (std.mem.eql(u8, typ, "filesystem.delete_file")) {
        const path = objectGetString(args, "path") orelse return error.MissingPath;
        app.fs_mutex.lock();
        defer app.fs_mutex.unlock();
        const resolved = try resolveWorkspacePath(app.allocator, app.config.workspace_root, path);
        defer app.allocator.free(resolved);
        try std.fs.cwd().deleteFile(resolved);
        var out = std.ArrayList(u8).init(app.allocator);
        errdefer out.deinit();
        const w = out.writer();
        try w.writeAll("{\"ok\":true,\"type\":\"filesystem.delete_file\",\"path\":");
        try writeJsonString(w, path);
        try w.print(",\"step\":{d},\"staleness\":{s},\"at\":{d}}}", .{ step, staleness_buf.items, now() });
        return out.toOwnedSlice();
    }
    if (std.mem.eql(u8, typ, "knowledge.query")) {
        const subject = objectGetString(args, "subject");
        const relation = objectGetString(args, "relation");
        const object = objectGetString(args, "object");
        app.graph_mutex.lock();
        defer app.graph_mutex.unlock();
        var results = try app.crev.queryInferenceKnowledge(subject, relation, object);
        defer results.deinit();
        var out = std.ArrayList(u8).init(app.allocator);
        errdefer out.deinit();
        const w = out.writer();
        try w.writeAll("{\"ok\":true,\"type\":\"knowledge.query\",\"count\":");
        try w.print("{}", .{results.items.len});
        try w.writeAll(",\"triplets\":[");
        for (results.items, 0..) |triplet, i| {
            if (i != 0) try w.writeAll(",");
            try w.writeAll("{\"subject\":");
            try writeJsonString(w, triplet.subject);
            try w.writeAll(",\"relation\":");
            try writeJsonString(w, triplet.relation);
            try w.writeAll(",\"object\":");
            try writeJsonString(w, triplet.object);
            try w.print(",\"confidence\":{d}}}", .{triplet.confidence});
        }
        try w.writeAll("]");
        try w.print(",\"step\":{d},\"staleness\":{s},\"at\":{d}}}", .{ step, staleness_buf.items, now() });
        return out.toOwnedSlice();
    }
    if (std.mem.eql(u8, typ, "http.get")) {
        const url = objectGetString(args, "url") orelse return error.MissingUrl;
        if (!isHttpHostAllowed(app.config.allowed_http_hosts, url)) return error.HttpHostNotAllowed;
        var client = std.http.Client{ .allocator = app.allocator };
        defer client.deinit();
        const uri = try std.Uri.parse(url);
        var header_buf: [4096]u8 = undefined;
        var req = try client.open(.GET, uri, .{ .server_header_buffer = &header_buf });
        defer req.deinit();
        try req.send();
        try req.finish();
        try req.wait();
        const body = try req.reader().readAllAlloc(app.allocator, 1024 * 1024);
        defer app.allocator.free(body);
        var out = std.ArrayList(u8).init(app.allocator);
        errdefer out.deinit();
        const w = out.writer();
        try w.writeAll("{\"ok\":true,\"type\":\"http.get\",\"url\":");
        try writeJsonString(w, url);
        try w.writeAll(",\"body\":");
        try writeJsonString(w, body);
        try w.print(",\"step\":{d},\"staleness\":{s},\"at\":{d}}}", .{ step, staleness_buf.items, now() });
        return out.toOwnedSlice();
    }
    return error.ActionNotAllowed;
}

fn routeTextThroughKnowledge(app: *App, run_id: []const u8, text: []const u8) void {
    if (text.len == 0) return;
    {
        app.graph_mutex.lock();
        defer app.graph_mutex.unlock();
        _ = app.crev.processTextStream(text) catch return;
    }
    refreshRunFacts(app, run_id) catch {};
}

fn validatedFactsJson(allocator: Allocator, index: *KnowledgeGraphIndex) ![]u8 {
    var out = std.ArrayList(u8).init(allocator);
    errdefer out.deinit();
    const w = out.writer();
    try w.writeAll("[");
    for (index.all_triplets.items, 0..) |triplet, i| {
        if (i != 0) try w.writeAll(",");
        try w.writeAll("{\"subject\":");
        try writeJsonString(w, triplet.subject);
        try w.writeAll(",\"relation\":");
        try writeJsonString(w, triplet.relation);
        try w.writeAll(",\"object\":");
        try writeJsonString(w, triplet.object);
        try w.print(",\"confidence\":{d}}}", .{triplet.confidence});
    }
    try w.writeAll("]");
    return out.toOwnedSlice();
}

fn refreshRunFacts(app: *App, run_id: []const u8) !void {
    const facts_json = blk: {
        app.graph_mutex.lock();
        defer app.graph_mutex.unlock();
        break :blk try validatedFactsJson(app.allocator, &app.crev.knowledge_index);
    };
    defer app.allocator.free(facts_json);
    try app.db.setRunStateKey(run_id, "facts", facts_json);
}

fn observationTextFromJson(allocator: Allocator, observation_json: []const u8) ![]u8 {
    var parsed = std.json.parseFromSlice(std.json.Value, allocator, observation_json, .{}) catch {
        return allocator.dupe(u8, observation_json);
    };
    defer parsed.deinit();
    if (objectGetString(parsed.value, "content")) |content| {
        return allocator.dupe(u8, content);
    }
    if (objectGetString(parsed.value, "text")) |text| {
        return allocator.dupe(u8, text);
    }
    return allocator.dupe(u8, observation_json);
}

fn evaluateStepVerifier(allocator: Allocator, state_json: []const u8, outcome_json: []const u8) ![]u8 {
    _ = state_json;
    const success = !containsIgnoreCase(outcome_json, "\"ok\":false") and !containsIgnoreCase(outcome_json, "\"error\":\"fatal\"");
    return std.fmt.allocPrint(allocator, "{{\"success\":{},\"checked_at\":{},\"scope\":\"step\"}}", .{ success, now() });
}

fn finalizeTrajectory(app: *App, run_id: []const u8) !void {
    if (!try app.db.claimFinalization(run_id)) return;
    var run = app.db.getRun(run_id) catch return;
    defer run.deinit(app.allocator);
    const verifier = try evaluateTerminalVerifiers(app, &run);
    defer app.allocator.free(verifier);
    _ = try app.db.insertEvent(run.tenant_id, run_id, run.step, "terminal_verifier", verifier);
    const trajectory_success = !containsIgnoreCase(verifier, "\"success\":false");
    calibrateRouterFromOutcome(app, run_id, trajectory_success);
    if (containsIgnoreCase(verifier, "\"success\":false")) {
        const reflection = generateReflectionPatch(app, &run, verifier) catch |err| blk: {
            break :blk try std.fmt.allocPrint(app.allocator, "{{\"diagnosis\":\"reflection_generation_failed\",\"error\":\"{s}\",\"patches\":[]}}", .{@errorName(err)});
        };
        defer app.allocator.free(reflection);
        _ = try app.db.insertEvent(run.tenant_id, run_id, run.step, "reflection_patch", reflection);
        const patch_id = try makeId(app.allocator, "patch");
        defer app.allocator.free(patch_id);
        const report = validateAndApplySkillPatch(app, run.tenant_id, reflection) catch |err| blk: {
            break :blk try std.fmt.allocPrint(app.allocator, "{{\"accepted\":false,\"error\":\"{s}\"}}", .{@errorName(err)});
        };
        defer app.allocator.free(report);
        try app.db.insertSkillPatch(run.tenant_id, patch_id, null, if (containsIgnoreCase(report, "\"accepted\":true")) "accepted" else "rejected", reflection, report);
        optimizePolicyFromReflection(app, &run, reflection) catch {};
    }
}

fn calibrateRouterFromOutcome(app: *App, run_id: []const u8, success: bool) void {
    const state_json = app.db.getStateJson(run_id) catch return;
    defer app.allocator.free(state_json);
    const observation_json = blk: {
        var run = app.db.getRun(run_id) catch return;
        defer run.deinit(app.allocator);
        break :blk app.allocator.dupe(u8, run.latest_observation_json) catch return;
    };
    defer app.allocator.free(observation_json);

    var combined = std.ArrayList(u8).init(app.allocator);
    defer combined.deinit();
    combined.writer().print("{s}\n{s}", .{ state_json, observation_json }) catch return;

    var query_tokens = std.ArrayList(u32).init(app.allocator);
    defer query_tokens.deinit();

    var samples = std.ArrayList([]u32).init(app.allocator);
    defer {
        for (samples.items) |sample| app.allocator.free(sample);
        samples.deinit();
    }
    var labels = std.ArrayList(f32).init(app.allocator);
    defer labels.deinit();

    const target: f32 = if (success) 1.0 else 0.0;

    app.retrieval_mutex.lock();
    defer app.retrieval_mutex.unlock();

    app.mgt.encode(combined.items, &query_tokens) catch return;
    if (query_tokens.items.len == 0) return;

    const retrieved = app.ssi.retrieveTopK(query_tokens.items, 8, app.allocator) catch return;
    defer freeRankedSegments(retrieved, app.allocator);
    for (retrieved) |segment| {
        const copy = app.allocator.dupe(u32, segment.tokens) catch return;
        samples.append(copy) catch {
            app.allocator.free(copy);
            return;
        };
        labels.append(target) catch return;
    }
    if (samples.items.len == 0) return;
    app.ranker.calibrateWeights(samples.items, labels.items, &app.ssi, 1) catch {};
}

fn evaluateTerminalVerifiers(app: *App, run: *RunRecord) ![]u8 {
    const state = try app.db.getStateJson(run.id);
    defer app.allocator.free(state);
    var parsed = std.json.parseFromSlice(std.json.Value, app.allocator, run.procedure_json, .{}) catch {
        const success = containsIgnoreCase(state, "\"done\":true") or std.mem.eql(u8, run.status, "completed");
        return std.fmt.allocPrint(app.allocator, "{{\"success\":{},\"scope\":\"terminal\",\"checked_at\":{}}}", .{ success, now() });
    };
    defer parsed.deinit();

    var all_passed = true;
    var details = std.ArrayList(u8).init(app.allocator);
    defer details.deinit();
    const dw = details.writer();
    try dw.writeAll("[");
    var has_verifiers = false;

    if (objectGetValue(parsed.value, "verifiers")) |verifiers| {
        switch (verifiers) {
            .array => |arr| {
                for (arr.items, 0..) |v, idx| {
                    has_verifiers = true;
                    if (idx != 0) try dw.writeAll(",");
                    const vtype = objectGetString(v, "type") orelse "unknown";
                    var passed = false;
                    if (std.mem.eql(u8, vtype, "state_key_exists")) {
                        if (objectGetString(v, "key")) |k| passed = stateHasKey(app.allocator, state, k);
                    } else if (std.mem.eql(u8, vtype, "state_key_equals")) {
                        if (objectGetString(v, "key")) |k| {
                            if (objectGetValue(v, "value")) |expected| passed = stateKeyEquals(app.allocator, state, k, expected);
                        }
                    } else if (std.mem.eql(u8, vtype, "file_exists")) {
                        if (objectGetString(v, "path")) |p| {
                            const resolved = resolveWorkspacePath(app.allocator, app.config.workspace_root, p) catch null;
                            if (resolved) |resolved_path| {
                                defer app.allocator.free(resolved_path);
                                passed = blk: {
                                    std.fs.cwd().access(resolved_path, .{}) catch break :blk false;
                                    break :blk true;
                                };
                            }
                        }
                    } else if (std.mem.eql(u8, vtype, "file_contains_lines")) {
                        if (objectGetString(v, "path")) |p| {
                            const lines = objectGetValue(v, "lines") orelse std.json.Value{ .array = std.json.Array.init(app.allocator) };
                            const resolved = resolveWorkspacePath(app.allocator, app.config.workspace_root, p) catch null;
                            if (resolved) |resolved_path| {
                                defer app.allocator.free(resolved_path);
                                var check_res_storage: []const u8 = "[]";
                                var check_res_owned = false;
                                if (checkLines(app.allocator, resolved_path, lines)) |check_result| {
                                    check_res_storage = check_result;
                                    check_res_owned = true;
                                } else |_| {}
                                passed = !containsIgnoreCase(check_res_storage, "false");
                                if (check_res_owned) app.allocator.free(check_res_storage);
                            }
                        }
                    } else if (std.mem.eql(u8, vtype, "min_progress")) {
                        const min_val = objectGetFloat(v, "value") orelse 1.0;
                        var state_parsed = std.json.parseFromSlice(std.json.Value, app.allocator, state, .{}) catch null;
                        if (state_parsed) |*sp| {
                            defer sp.deinit();
                            const actual_prog = objectGetFloat(sp.value, "progress") orelse 0.0;
                            passed = actual_prog >= min_val;
                        }
                    }
                    if (!passed) all_passed = false;
                    try dw.writeAll("{\"type\":");
                    try writeJsonString(dw, vtype);
                    try dw.print(",\"passed\":{}}}", .{passed});
                }
            },
            else => {},
        }
    }
    try dw.writeAll("]");

    if (!has_verifiers) {
        all_passed = containsIgnoreCase(state, "\"done\":true") or std.mem.eql(u8, run.status, "completed");
    }

    return std.fmt.allocPrint(app.allocator, "{{\"success\":{},\"scope\":\"terminal\",\"details\":{s},\"checked_at\":{}}}", .{ all_passed, details.items, now() });
}

fn stateHasKey(allocator: Allocator, state_json: []const u8, key: []const u8) bool {
    var parsed = std.json.parseFromSlice(std.json.Value, allocator, state_json, .{}) catch return false;
    defer parsed.deinit();
    return objectGetValue(parsed.value, key) != null;
}

fn stateKeyEquals(allocator: Allocator, state_json: []const u8, key: []const u8, expected: std.json.Value) bool {
    var parsed = std.json.parseFromSlice(std.json.Value, allocator, state_json, .{}) catch return false;
    defer parsed.deinit();
    const actual = objectGetValue(parsed.value, key) orelse return false;
    const a = jsonValueToOwned(allocator, actual) catch return false;
    defer allocator.free(a);
    const e = jsonValueToOwned(allocator, expected) catch return false;
    defer allocator.free(e);
    return std.mem.eql(u8, a, e);
}

fn generateReflectionPatch(app: *App, run: *RunRecord, verifier_json: []const u8) ![]u8 {
    const state = try app.db.getStateJson(run.id);
    defer app.allocator.free(state);
    var prompt = std.ArrayList(u8).init(app.allocator);
    defer prompt.deinit();
    try prompt.writer().print("Generate compact JSON reflection patch. Output JSON only with keys diagnosis, failure_component, pivot_actions, skill_patch. Procedure: {s}\nTerminal state: {s}\nVerifier: {s}", .{ run.procedure_json, state, verifier_json });
    var result = try callModelStreaming(app, prompt.items);
    defer result.deinit(app.allocator);
    return extractJsonObject(app.allocator, result.content);
}

fn validateAndApplySkillPatch(app: *App, tenant_id: []const u8, patch_json: []const u8) ![]u8 {
    var parsed = try std.json.parseFromSlice(std.json.Value, app.allocator, patch_json, .{});
    defer parsed.deinit();
    try validateValueBounded(parsed.value, 0);
    const skill_patch = objectGetValue(parsed.value, "skill_patch") orelse parsed.value;
    const skill = objectGetValue(skill_patch, "skill") orelse return error.MissingSkill;
    const skill_id = if (objectGetString(skill, "id")) |sid| try app.allocator.dupe(u8, sid) else try makeId(app.allocator, "skill");
    defer app.allocator.free(skill_id);
    const name = objectGetString(skill, "name") orelse return error.SkillMissingName;
    const description = objectGetString(skill, "description") orelse return error.SkillMissingDescription;
    const trigger_value = objectGetValue(skill, "trigger") orelse std.json.Value{ .object = std.json.ObjectMap.init(app.allocator) };
    const procedure_value = objectGetValue(skill, "procedure") orelse return error.SkillMissingProcedure;
    const trigger_json = try jsonValueToOwned(app.allocator, trigger_value);
    defer app.allocator.free(trigger_json);
    const procedure_json = try jsonValueToOwned(app.allocator, procedure_value);
    defer app.allocator.free(procedure_json);

    try registerSkill(app, tenant_id, skill_id, name, description, trigger_json, procedure_json, true);
    var out = std.ArrayList(u8).init(app.allocator);
    errdefer out.deinit();
    const w = out.writer();
    try w.writeAll("{\"accepted\":true,\"skill_id\":");
    try writeJsonString(w, skill_id);
    try w.print(",\"validated_at\":{d}}}", .{now()});
    return out.toOwnedSlice();
}

fn optimizePolicyFromReflection(app: *App, run: *RunRecord, reflection_json: []const u8) !void {
    const state = try app.db.getStateJson(run.id);
    defer app.allocator.free(state);
    const clean_prompt = try buildStepPrompt(app.allocator, run.procedure_json, state, run.latest_observation_json, "[]");
    defer app.allocator.free(clean_prompt);
    const privileged_prompt = try std.fmt.allocPrint(app.allocator, "{{\"reflection_patch\":{s},\"clean_prompt\":{s}}}", .{ reflection_json, clean_prompt });
    defer app.allocator.free(privileged_prompt);
    const teacher = try callModelLogprobs(app, privileged_prompt);
    defer app.allocator.free(teacher);
    const student = try callModelLogprobs(app, clean_prompt);
    defer app.allocator.free(student);
    const kl = computeReverseKl(app.allocator, teacher, student) catch 0.0;
    try app.db.insertDistillationExample(run.tenant_id, run.id, reflection_json, clean_prompt, teacher, student, kl);
}

fn callModelLogprobs(app: *App, user_prompt: []const u8) ![]u8 {
    var client = std.http.Client{ .allocator = app.allocator };
    defer client.deinit();
    const url = try joinUrl(app.allocator, app.config.modular_base_url, "chat/completions");
    defer app.allocator.free(url);
    const uri = try std.Uri.parse(url);

    var body_buf = std.ArrayList(u8).init(app.allocator);
    defer body_buf.deinit();
    const bw = body_buf.writer();
    try bw.writeAll("{\"model\":");
    try writeJsonString(bw, app.config.model);
    try bw.writeAll(",\"messages\":[{\"role\":\"user\",\"content\":");
    try writeJsonString(bw, user_prompt);
    try bw.writeAll("}],\"stream\":false,\"temperature\":0.7,\"max_tokens\":2048,\"logprobs\":true,\"top_logprobs\":5,\"response_format\":{\"type\":\"json_object\"}}");

    const auth = try std.fmt.allocPrint(app.allocator, "Bearer {s}", .{app.config.modular_api_key});
    defer app.allocator.free(auth);

    var header_buf: [8192]u8 = undefined;
    var req = try client.open(.POST, uri, .{
        .server_header_buffer = &header_buf,
        .extra_headers = &.{
            .{ .name = "authorization", .value = auth },
            .{ .name = "content-type", .value = "application/json" },
        },
    });
    defer req.deinit();

    req.transfer_encoding = .{ .content_length = body_buf.items.len };
    try req.send();
    try req.writeAll(body_buf.items);
    try req.finish();
    try req.wait();

    if (req.response.status != .ok) return error.ModelRequestFailed;
    const body = try req.reader().readAllAlloc(app.allocator, app.config.max_response_bytes);
    defer app.allocator.free(body);

    var parsed = try std.json.parseFromSlice(std.json.Value, app.allocator, body, .{});
    defer parsed.deinit();
    if (objectGetValue(parsed.value, "choices")) |choices| switch (choices) {
        .array => |arr| {
            if (arr.items.len > 0) {
                if (objectGetValue(arr.items[0], "logprobs")) |lp| return jsonValueToOwned(app.allocator, lp);
            }
        },
        else => {},
    };
    return app.allocator.dupe(u8, "{}");
}

fn computeReverseKl(allocator: Allocator, teacher_json: []const u8, student_json: []const u8) !f64 {
    var teacher = try std.json.parseFromSlice(std.json.Value, allocator, teacher_json, .{});
    defer teacher.deinit();
    var student = try std.json.parseFromSlice(std.json.Value, allocator, student_json, .{});
    defer student.deinit();
    const tc = objectGetValue(teacher.value, "content") orelse return 0.0;
    const sc = objectGetValue(student.value, "content") orelse return 0.0;
    var kl: f64 = 0.0;
    switch (tc) {
        .array => |ta| switch (sc) {
            .array => |sa| {
                const n = @min(ta.items.len, sa.items.len);
                var i: usize = 0;
                while (i < n) : (i += 1) {
                    const tops = objectGetValue(ta.items[i], "top_logprobs") orelse continue;
                    switch (tops) {
                        .array => |toparr| {
                            for (toparr.items) |top| {
                                const tok = objectGetString(top, "token") orelse continue;
                                const logq = objectGetFloat(top, "logprob") orelse continue;
                                const logp = findLogprob(sa.items[i], tok) orelse -30.0;
                                const q = std.math.exp(logq);
                                kl += q * (logq - logp);
                            }
                        },
                        else => {},
                    }
                }
            },
            else => {},
        },
        else => {},
    }
    return kl;
}

fn findLogprob(item: std.json.Value, token: []const u8) ?f64 {
    const tops = objectGetValue(item, "top_logprobs") orelse return null;
    switch (tops) {
        .array => |arr| {
            for (arr.items) |top| {
                const tok = objectGetString(top, "token") orelse continue;
                if (std.mem.eql(u8, tok, token)) return objectGetFloat(top, "logprob");
            }
        },
        else => {},
    }
    return null;
}

fn metaHarnessEntry(app: *App) void {
    while (true) {
        consolidateKnowledge(app, app.config.default_tenant_id) catch {};
        std.time.sleep(300 * std.time.ns_per_s);
    }
}

fn consolidateKnowledge(app: *App, tenant_id: []const u8) !void {
    const skills_markdown = try app.db.buildKnowledgeMarkdown(tenant_id);
    defer app.allocator.free(skills_markdown);
    const facts_markdown = try buildKnowledgeFactsMarkdown(app);
    defer app.allocator.free(facts_markdown);
    var full_markdown = std.ArrayList(u8).init(app.allocator);
    defer full_markdown.deinit();
    try full_markdown.appendSlice(skills_markdown);
    try full_markdown.appendSlice(facts_markdown);
    const markdown = try full_markdown.toOwnedSlice();
    defer app.allocator.free(markdown);
    try std.fs.cwd().makePath(app.config.knowledge_root);
    const playbook_path = try std.fs.path.join(app.allocator, &.{ app.config.knowledge_root, "PLAYBOOK.md" });
    defer app.allocator.free(playbook_path);
    var old_storage: []const u8 = "";
    var old_owned = false;
    if (std.fs.cwd().readFileAlloc(app.allocator, playbook_path, 16 * 1024 * 1024)) |loaded| {
        old_storage = loaded;
        old_owned = true;
    } else |_| {}
    defer if (old_owned) app.allocator.free(old_storage);
    try atomicWriteFile(app.allocator, playbook_path, markdown);
    _ = runGit(app, &[_][]const u8{ "git", "-C", app.config.knowledge_root, "init" }) catch {};
    _ = runGit(app, &[_][]const u8{ "git", "-C", app.config.knowledge_root, "add", "PLAYBOOK.md" }) catch {};
    const message = try std.fmt.allocPrint(app.allocator, "knowledge consolidation {}", .{now()});
    defer app.allocator.free(message);
    _ = runGit(app, &[_][]const u8{ "git", "-C", app.config.knowledge_root, "-c", "user.name=agent-runtime", "-c", "user.email=agent-runtime@local", "commit", "-m", message }) catch {};
    const commit = runGit(app, &[_][]const u8{ "git", "-C", app.config.knowledge_root, "rev-parse", "HEAD" }) catch try app.allocator.dupe(u8, "head");
    defer app.allocator.free(commit);
    const diff = diffText(app.allocator, old_storage, markdown) catch try app.allocator.dupe(u8, "no diff");
    defer app.allocator.free(diff);
    try app.db.insertKnowledgeVersion(tenant_id, markdown, std.mem.trim(u8, commit, " \t\r\n"), diff);
}

fn buildKnowledgeFactsMarkdown(app: *App) ![]u8 {
    app.graph_mutex.lock();
    defer app.graph_mutex.unlock();
    var out = std.ArrayList(u8).init(app.allocator);
    errdefer out.deinit();
    const w = out.writer();
    try w.writeAll("\n## Verified Knowledge Graph Facts\n\n");
    const total = app.crev.knowledge_index.all_triplets.items.len;
    if (total == 0) {
        try w.writeAll("No validated relational facts recorded yet.\n");
        return out.toOwnedSlice();
    }
    var listed: usize = 0;
    for (app.crev.knowledge_index.all_triplets.items) |triplet| {
        if (listed >= 200) break;
        listed += 1;
        try w.print("- {s} --[{s}]--> {s} (confidence {d:.4})\n", .{ triplet.subject, triplet.relation, triplet.object, triplet.confidence });
    }
    if (total > listed) {
        try w.print("\n...and {} more validated facts held in the relational knowledge graph.\n", .{total - listed});
    }
    return out.toOwnedSlice();
}

fn runGit(app: *App, argv: []const []const u8) ![]u8 {
    const result = try std.process.Child.run(.{ .allocator = app.allocator, .argv = argv, .max_output_bytes = 1024 * 1024 });
    defer app.allocator.free(result.stderr);
    switch (result.term) {
        .Exited => |code| {
            if (code != 0) {
                app.allocator.free(result.stdout);
                return error.GitFailed;
            }
        },
        else => {
            app.allocator.free(result.stdout);
            return error.GitFailed;
        },
    }
    return result.stdout;
}

const DiffOp = struct {
    const Kind = enum { remove, insert };
    kind: Kind,
    line_no: usize,
    text: []const u8,
};

fn diffText(allocator: Allocator, old: []const u8, new: []const u8) ![]u8 {
    var out = std.ArrayList(u8).init(allocator);
    errdefer out.deinit();
    if (std.mem.eql(u8, old, new)) {
        try out.writer().writeAll("no changes\n");
        return out.toOwnedSlice();
    }
    var old_lines = std.ArrayList([]const u8).init(allocator);
    defer old_lines.deinit();
    var old_split = std.mem.splitScalar(u8, old, '\n');
    while (old_split.next()) |line| try old_lines.append(line);
    var new_lines = std.ArrayList([]const u8).init(allocator);
    defer new_lines.deinit();
    var new_split = std.mem.splitScalar(u8, new, '\n');
    while (new_split.next()) |line| try new_lines.append(line);

    const m = old_lines.items.len;
    const n = new_lines.items.len;
    const row_stride = n + 1;
    const cell_count = std.math.mul(usize, m + 1, row_stride) catch return error.DiffTooLarge;
    var table = try allocator.alloc(u32, cell_count);
    defer allocator.free(table);
    @memset(table, 0);

    var i: usize = 1;
    while (i <= m) : (i += 1) {
        var j: usize = 1;
        while (j <= n) : (j += 1) {
            if (std.mem.eql(u8, old_lines.items[i - 1], new_lines.items[j - 1])) {
                table[i * row_stride + j] = table[(i - 1) * row_stride + (j - 1)] + 1;
            } else {
                const up = table[(i - 1) * row_stride + j];
                const left = table[i * row_stride + (j - 1)];
                table[i * row_stride + j] = if (up >= left) up else left;
            }
        }
    }

    var ops = std.ArrayList(DiffOp).init(allocator);
    defer ops.deinit();
    var bi = m;
    var bj = n;
    while (bi > 0 or bj > 0) {
        if (bi > 0 and bj > 0 and std.mem.eql(u8, old_lines.items[bi - 1], new_lines.items[bj - 1])) {
            bi -= 1;
            bj -= 1;
        } else if (bj > 0 and (bi == 0 or table[bi * row_stride + (bj - 1)] >= table[(bi - 1) * row_stride + bj])) {
            try ops.append(.{ .kind = .insert, .line_no = bj, .text = new_lines.items[bj - 1] });
            bj -= 1;
        } else {
            try ops.append(.{ .kind = .remove, .line_no = bi, .text = old_lines.items[bi - 1] });
            bi -= 1;
        }
    }

    var index: usize = ops.items.len;
    while (index > 0) {
        index -= 1;
        const op = ops.items[index];
        switch (op.kind) {
            .remove => try out.writer().print("-{}:{s}\n", .{ op.line_no, op.text }),
            .insert => try out.writer().print("+{}:{s}\n", .{ op.line_no, op.text }),
        }
    }
    return out.toOwnedSlice();
}

fn startHttpServer(app: *App) !void {
    const address = try std.net.Address.parseIp(app.config.host, app.config.port);
    const fd = try std.posix.socket(std.posix.AF.INET, std.posix.SOCK.STREAM, 0);
    defer std.posix.close(fd);
    var yes: c_int = 1;
    try std.posix.setsockopt(fd, std.posix.SOL.SOCKET, std.posix.SO.REUSEADDR, std.mem.asBytes(&yes));
    try std.posix.bind(fd, &address.any, address.getOsSockLen());
    try std.posix.listen(fd, 128);
    while (true) {
        const client = try std.posix.accept(fd, null, null, 0);
        const thread = try std.Thread.spawn(.{}, connectionEntry, .{ app, client });
        thread.detach();
    }
}

fn connectionEntry(app: *App, fd: std.posix.socket_t) void {
    defer std.posix.close(fd);
    var arena = std.heap.ArenaAllocator.init(app.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    var req = readHttpRequest(allocator, fd, app.config.max_request_bytes) catch return;
    handleRequest(app, allocator, fd, &req) catch return;
}

fn readHttpRequest(allocator: Allocator, fd: std.posix.socket_t, max_bytes: usize) !HttpRequest {
    var buf = std.ArrayList(u8).init(allocator);
    var temp: [8192]u8 = undefined;
    var header_end: ?usize = null;
    var content_length: usize = 0;
    while (true) {
        const n = try std.posix.read(fd, &temp);
        if (n == 0) break;
        try buf.appendSlice(temp[0..n]);
        if (buf.items.len > max_bytes) return error.RequestTooLarge;
        if (header_end == null) {
            if (std.mem.indexOf(u8, buf.items, "\r\n\r\n")) |idx| {
                header_end = idx;
                content_length = parseContentLength(buf.items[0..idx]);
            }
        }
        if (header_end) |he| {
            if (buf.items.len >= he + 4 + content_length) break;
        }
    }
    const raw = buf.items;
    const he = header_end orelse return error.BadRequest;
    const headers_part = raw[0..he];
    const body_start = he + 4;
    if (raw.len < body_start + content_length) return error.BadRequest;
    const body = raw[body_start .. body_start + content_length];
    var lines = std.mem.splitSequence(u8, headers_part, "\r\n");
    const start_line = lines.next() orelse return error.BadRequest;
    var parts = std.mem.tokenizeScalar(u8, start_line, ' ');
    const method = parts.next() orelse return error.BadRequest;
    const target = parts.next() orelse return error.BadRequest;
    var path = target;
    var query: []const u8 = "";
    if (std.mem.indexOfScalar(u8, target, '?')) |qidx| {
        path = target[0..qidx];
        query = target[qidx + 1 ..];
    }
    var headers = std.StringHashMap([]const u8).init(allocator);
    while (lines.next()) |line| {
        if (std.mem.indexOfScalar(u8, line, ':')) |colon| {
            const key_raw = std.mem.trim(u8, line[0..colon], " \t");
            const value = std.mem.trim(u8, line[colon + 1 ..], " \t");
            const key = try allocator.alloc(u8, key_raw.len);
            for (key_raw, 0..) |c, i| key[i] = std.ascii.toLower(c);
            try headers.put(key, value);
        }
    }
    return HttpRequest{ .method = method, .target = target, .path = path, .query = query, .headers = headers, .body = body };
}

fn parseContentLength(headers_part: []const u8) usize {
    var lines = std.mem.splitSequence(u8, headers_part, "\r\n");
    _ = lines.next();
    while (lines.next()) |line| {
        if (std.mem.indexOfScalar(u8, line, ':')) |colon| {
            const key = std.mem.trim(u8, line[0..colon], " \t");
            if (std.ascii.eqlIgnoreCase(key, "content-length")) {
                const value = std.mem.trim(u8, line[colon + 1 ..], " \t");
                return std.fmt.parseInt(usize, value, 10) catch 0;
            }
        }
    }
    return 0;
}

fn handleRequest(app: *App, allocator: Allocator, fd: std.posix.socket_t, req: *HttpRequest) !void {
    if (std.mem.eql(u8, req.method, "OPTIONS")) {
        try sendResponse(allocator, fd, 204, "text/plain", "");
        return;
    }
    const tenant_id = req.headers.get("x-tenant-id") orelse app.config.default_tenant_id;
    try app.db.ensureTenant(tenant_id);

    if (std.mem.eql(u8, req.method, "GET") and (std.mem.eql(u8, req.path, "/") or std.mem.eql(u8, req.path, "/index.html"))) {
        const index = std.fs.cwd().readFileAlloc(allocator, "index.html", 16 * 1024 * 1024) catch "<!doctype html><html><body><h1>Autonomous Agent Runtime</h1></body></html>";
        try sendResponse(allocator, fd, 200, "text/html; charset=utf-8", index);
        return;
    }
    if (std.mem.eql(u8, req.method, "GET") and std.mem.eql(u8, req.path, "/api/health")) {
        try sendResponse(allocator, fd, 200, "application/json", "{\"ok\":true,\"runtime\":\"autonomous-state-agent\"}");
        return;
    }
    if (std.mem.eql(u8, req.method, "POST") and (std.mem.eql(u8, req.path, "/api/runs") or std.mem.eql(u8, req.path, "/api/tasks") or std.mem.eql(u8, req.path, "/api/chat"))) {
        const response = try createRunOrChat(app, allocator, tenant_id, req.body);
        try sendResponse(allocator, fd, 200, "application/json", response);
        return;
    }
    if (std.mem.eql(u8, req.method, "GET") and std.mem.eql(u8, req.path, "/api/runs")) {
        const json = try app.db.listRunsJson(tenant_id);
        defer app.allocator.free(json);
        try sendResponse(allocator, fd, 200, "application/json", json);
        return;
    }
    if (std.mem.startsWith(u8, req.path, "/api/runs/")) {
        const rest = req.path["/api/runs/".len..];
        var segs = std.mem.splitScalar(u8, rest, '/');
        const run_id = segs.next() orelse return error.BadRequest;
        const tail = segs.next();
        if (tail == null and std.mem.eql(u8, req.method, "GET")) {
            const json = try app.db.runJson(tenant_id, run_id);
            defer app.allocator.free(json);
            try sendResponse(allocator, fd, 200, "application/json", json);
            return;
        }
        if (tail != null and std.mem.eql(u8, tail.?, "state") and std.mem.eql(u8, req.method, "GET")) {
            const state = try app.db.getStateJson(run_id);
            defer app.allocator.free(state);
            try sendResponse(allocator, fd, 200, "application/json", state);
            return;
        }
        if (tail != null and std.mem.eql(u8, tail.?, "events") and std.mem.eql(u8, req.method, "GET")) {
            try streamEvents(app, allocator, fd, tenant_id, run_id, req.query);
            return;
        }
        if (tail != null and std.mem.eql(u8, tail.?, "observe") and std.mem.eql(u8, req.method, "POST")) {
            try validateJsonText(allocator, req.body);
            try app.db.enqueueObservation(run_id, req.body);
            _ = try app.db.insertEvent(tenant_id, run_id, 0, "external_observation", req.body);
            const obs_text = try observationTextFromJson(allocator, req.body);
            defer allocator.free(obs_text);
            routeTextThroughKnowledge(app, run_id, obs_text);
            try sendResponse(allocator, fd, 200, "application/json", "{\"ok\":true}");
            return;
        }
        if (tail != null and std.mem.eql(u8, tail.?, "stop") and std.mem.eql(u8, req.method, "POST")) {
            try app.db.markRunStatus(run_id, "stopped", null);
            try sendResponse(allocator, fd, 200, "application/json", "{\"ok\":true}");
            return;
        }
        if (tail != null and std.mem.eql(u8, tail.?, "resume") and std.mem.eql(u8, req.method, "POST")) {
            try app.db.markRunStatus(run_id, "running", null);
            try spawnRun(app, run_id);
            try sendResponse(allocator, fd, 200, "application/json", "{\"ok\":true}");
            return;
        }
    }
    if (std.mem.eql(u8, req.method, "GET") and std.mem.eql(u8, req.path, "/api/skills/search")) {
        const q = try queryParam(allocator, req.query, "q");
        const skills = try routeSkills(app, tenant_id, q, "{}");
        defer freeSkillRecords(app.allocator, skills);
        const json = try selectedSkillsJson(allocator, skills);
        try sendResponse(allocator, fd, 200, "application/json", json);
        return;
    }
    if (std.mem.eql(u8, req.method, "POST") and std.mem.eql(u8, req.path, "/api/meta/consolidate")) {
        try consolidateKnowledge(app, tenant_id);
        try sendResponse(allocator, fd, 200, "application/json", "{\"ok\":true}");
        return;
    }
    try sendResponse(allocator, fd, 404, "application/json", "{\"error\":\"not_found\"}");
}

fn createRunOrChat(app: *App, allocator: Allocator, tenant_id: []const u8, body: []const u8) ![]u8 {
    var parsed = std.json.parseFromSlice(std.json.Value, allocator, body, .{}) catch {
        const escaped_task = try jsonStringAlloc(allocator, body);
        const procedure = try std.fmt.allocPrint(allocator, "{{\"objective\":{s},\"max_steps\":{}}}", .{ escaped_task, app.config.max_steps });
        const initial = try std.fmt.allocPrint(allocator, "{{\"type\":\"user_task\",\"content\":{s}}}", .{escaped_task});
        const run_id = try makeId(app.allocator, "run");
        try app.db.createRun(tenant_id, run_id, procedure, initial, app.config.default_token_budget);
        try spawnRun(app, run_id);
        const run_json = try jsonStringAlloc(allocator, run_id);
        app.allocator.free(run_id);
        return std.fmt.allocPrint(allocator, "{{\"ok\":true,\"run_id\":{s},\"status\":\"running\"}}", .{run_json});
    };
    defer parsed.deinit();
    const task = objectGetString(parsed.value, "task") orelse objectGetString(parsed.value, "message") orelse "task";
    const task_json = try jsonStringAlloc(allocator, task);
    const procedure_json = try std.fmt.allocPrint(allocator, "{{\"objective\":{s},\"max_steps\":{}}}", .{ task_json, app.config.max_steps });
    const initial_observation = try std.fmt.allocPrint(allocator, "{{\"type\":\"user_task\",\"content\":{s}}}", .{task_json});
    const run_id = try makeId(app.allocator, "run");
    try app.db.createRun(tenant_id, run_id, procedure_json, initial_observation, app.config.default_token_budget);
    try spawnRun(app, run_id);
    const run_id_json = try jsonStringAlloc(allocator, run_id);
    app.allocator.free(run_id);
    return std.fmt.allocPrint(allocator, "{{\"ok\":true,\"run_id\":{s},\"status\":\"running\"}}", .{run_id_json});
}

fn streamEvents(app: *App, allocator: Allocator, fd: std.posix.socket_t, tenant_id: []const u8, run_id: []const u8, query: []const u8) !void {
    const header = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\nAccess-Control-Allow-Origin: *\r\n\r\n";
    try writeAllFd(fd, header);
    const last_id_text = queryParam(allocator, query, "last_id") catch "";
    var last_id = std.fmt.parseInt(i64, last_id_text, 10) catch 0;
    var loops: usize = 0;
    while (loops < 3600) : (loops += 1) {
        const events = try app.db.eventsSinceJson(tenant_id, run_id, last_id);
        defer app.allocator.free(events);
        var parsed = try std.json.parseFromSlice(std.json.Value, allocator, events, .{});
        defer parsed.deinit();
        switch (parsed.value) {
            .array => |arr| {
                for (arr.items) |event| {
                    const id = objectGetInt(event, "id") orelse last_id;
                    const typ = objectGetString(event, "type") orelse "message";
                    const payload = objectGetValue(event, "payload") orelse event;
                    const payload_json = try jsonValueToOwned(allocator, payload);
                    defer allocator.free(payload_json);
                    const chunk = try std.fmt.allocPrint(allocator, "id: {}\nevent: {s}\ndata: {s}\n\n", .{ id, typ, payload_json });
                    defer allocator.free(chunk);
                    try writeAllFd(fd, chunk);
                    last_id = id;
                }
            },
            else => {},
        }
        try writeAllFd(fd, ": keepalive\n\n");
        std.time.sleep(1000 * std.time.ns_per_ms);
    }
}

fn sendResponse(allocator: Allocator, fd: std.posix.socket_t, status: u16, content_type: []const u8, body: []const u8) !void {
    const reason = switch (status) {
        200 => "OK",
        204 => "No Content",
        400 => "Bad Request",
        404 => "Not Found",
        else => "OK",
    };
    const header = try std.fmt.allocPrint(allocator, "HTTP/1.1 {} {s}\r\nContent-Type: {s}\r\nContent-Length: {}\r\nConnection: close\r\nAccess-Control-Allow-Origin: *\r\n\r\n", .{ status, reason, content_type, body.len });
    defer allocator.free(header);
    try writeAllFd(fd, header);
    try writeAllFd(fd, body);
}

fn writeAllFd(fd: std.posix.socket_t, data: []const u8) !void {
    var offset: usize = 0;
    while (offset < data.len) {
        const n = try std.posix.write(fd, data[offset..]);
        if (n == 0) return error.ConnectionClosed;
        offset += n;
    }
}

fn queryParam(allocator: Allocator, query: []const u8, name: []const u8) ![]u8 {
    var pairs = std.mem.splitScalar(u8, query, '&');
    while (pairs.next()) |pair| {
        if (std.mem.indexOfScalar(u8, pair, '=')) |eq| {
            if (std.mem.eql(u8, pair[0..eq], name)) return allocator.dupe(u8, pair[eq + 1 ..]);
        }
    }
    return allocator.dupe(u8, "");
}

fn writeJsonString(writer: anytype, s: []const u8) !void {
    try writer.writeByte('"');
    for (s) |ch| {
        switch (ch) {
            '"' => try writer.writeAll("\\\""),
            '\\' => try writer.writeAll("\\\\"),
            '\n' => try writer.writeAll("\\n"),
            '\r' => try writer.writeAll("\\r"),
            '\t' => try writer.writeAll("\\t"),
            0...7, 11, 12, 14...31 => try writer.print("\\u{X:0>4}", .{ch}),
            else => try writer.writeByte(ch),
        }
    }
    try writer.writeByte('"');
}

fn jsonStringAlloc(allocator: Allocator, s: []const u8) ![]u8 {
    var out = std.ArrayList(u8).init(allocator);
    try writeJsonString(out.writer(), s);
    return out.toOwnedSlice();
}

fn jsonValueToOwned(allocator: Allocator, value: std.json.Value) ![]u8 {
    var out = std.ArrayList(u8).init(allocator);
    try std.json.stringify(value, .{}, out.writer());
    return out.toOwnedSlice();
}

fn validateJsonText(allocator: Allocator, text: []const u8) !void {
    var parsed = try std.json.parseFromSlice(std.json.Value, allocator, text, .{});
    defer parsed.deinit();
    try validateValueBounded(parsed.value, 0);
}

fn validateValueBounded(value: std.json.Value, depth: usize) !void {
    if (depth > 12) return error.JsonTooDeep;
    switch (value) {
        .object => |obj| {
            var count: usize = 0;
            var it = obj.iterator();
            while (it.next()) |entry| {
                count += 1;
                if (count > 256) return error.JsonObjectTooLarge;
                if (entry.key_ptr.*.len > 256) return error.JsonKeyTooLarge;
                if (containsBannedKey(entry.key_ptr.*)) return error.BannedStateKey;
                try validateValueBounded(entry.value_ptr.*, depth + 1);
            }
        },
        .array => |arr| {
            if (arr.items.len > 128) return error.JsonArrayTooLarge;
            for (arr.items) |item| try validateValueBounded(item, depth + 1);
        },
        .string => |s| {
            if (s.len > 32768) return error.JsonStringTooLarge;
            if (containsIgnoreCase(s, "chain_of_thought") or containsIgnoreCase(s, "reasoning trace") or containsIgnoreCase(s, "reasoning_trace")) return error.ReasoningLeak;
        },
        else => {},
    }
}

fn validatePatchValue(value: std.json.Value) !void {
    switch (value) {
        .object => {},
        else => return error.InvalidStatePatch,
    }
    try validateValueBounded(value, 0);
}

fn containsIgnoreCase(haystack: []const u8, needle: []const u8) bool {
    if (needle.len == 0) return true;
    if (haystack.len < needle.len) return false;
    var i: usize = 0;
    while (i + needle.len <= haystack.len) : (i += 1) {
        var j: usize = 0;
        while (j < needle.len) : (j += 1) {
            if (std.ascii.toLower(haystack[i + j]) != std.ascii.toLower(needle[j])) break;
        }
        if (j == needle.len) return true;
    }
    return false;
}

fn containsBannedKey(key: []const u8) bool {
    return std.ascii.eqlIgnoreCase(key, "history") or
        std.ascii.eqlIgnoreCase(key, "messages") or
        std.ascii.eqlIgnoreCase(key, "transcript") or
        std.ascii.eqlIgnoreCase(key, "chain_of_thought") or
        std.ascii.eqlIgnoreCase(key, "reasoning_trace") or
        std.ascii.eqlIgnoreCase(key, "scratchpad") or
        std.ascii.eqlIgnoreCase(key, "prior_actions") or
        std.ascii.eqlIgnoreCase(key, "past_reasoning") or
        std.ascii.eqlIgnoreCase(key, "raw_dialogue");
}

fn estimateTokens(bytes: usize) i64 {
    const v = (bytes + 3) / 4;
    return @intCast(v);
}

fn objectGetValue(value: std.json.Value, key: []const u8) ?std.json.Value {
    switch (value) {
        .object => |obj| return obj.get(key),
        else => return null,
    }
}

fn objectGetString(value: std.json.Value, key: []const u8) ?[]const u8 {
    const v = objectGetValue(value, key) orelse return null;
    switch (v) {
        .string => |s| return s,
        else => return null,
    }
}

fn objectGetBool(value: std.json.Value, key: []const u8) ?bool {
    const v = objectGetValue(value, key) orelse return null;
    switch (v) {
        .bool => |b| return b,
        else => return null,
    }
}

fn objectGetInt(value: std.json.Value, key: []const u8) ?i64 {
    const v = objectGetValue(value, key) orelse return null;
    switch (v) {
        .integer => |i| return i,
        .float => |f| return @intFromFloat(f),
        else => return null,
    }
}

fn objectGetFloat(value: std.json.Value, key: []const u8) ?f64 {
    const v = objectGetValue(value, key) orelse return null;
    switch (v) {
        .integer => |i| return @floatFromInt(i),
        .float => |f| return f,
        else => return null,
    }
}

fn makeFtsQuery(allocator: Allocator, text: []const u8) ![]u8 {
    var out = std.ArrayList(u8).init(allocator);
    var hash_set = std.StringHashMap(void).init(allocator);
    var token = std.ArrayList(u8).init(allocator);
    defer token.deinit();
    var emitted: usize = 0;
    for (text) |c0| {
        const c = std.ascii.toLower(c0);
        if (std.ascii.isAlphanumeric(c)) {
            if (token.items.len < 32) try token.append(c);
        } else {
            if (token.items.len >= 3 and emitted < 12 and !hash_set.contains(token.items)) {
                const copy = try allocator.dupe(u8, token.items);
                try hash_set.put(copy, {});
                if (emitted != 0) try out.writer().writeAll(" OR ");
                try out.writer().print("{s}", .{token.items});
                emitted += 1;
            }
            token.clearRetainingCapacity();
        }
    }
    if (token.items.len >= 3 and emitted < 12 and !hash_set.contains(token.items)) {
        const copy = try allocator.dupe(u8, token.items);
        try hash_set.put(copy, {});
        if (emitted != 0) try out.writer().writeAll(" OR ");
        try out.writer().print("{s}", .{token.items});
        emitted += 1;
    }
    var it = hash_set.iterator();
    while (it.next()) |entry| allocator.free(entry.key_ptr.*);
    hash_set.deinit();
    return out.toOwnedSlice();
}

fn resolveWorkspacePath(allocator: Allocator, root: []const u8, rel: []const u8) ![]u8 {
    if (rel.len == 0) return error.InvalidPath;
    if (std.mem.startsWith(u8, rel, "/") or std.mem.indexOfScalar(u8, rel, '\\') != null or std.mem.indexOfScalar(u8, rel, 0) != null or std.mem.indexOfScalar(u8, rel, ':') != null) return error.UnauthorizedPath;
    var parts = std.mem.splitScalar(u8, rel, '/');
    while (parts.next()) |part| {
        if (part.len == 0 or std.mem.eql(u8, part, ".") or std.mem.eql(u8, part, "..")) return error.UnauthorizedPath;
    }
    try std.fs.cwd().makePath(root);
    const root_real = try std.fs.cwd().realpathAlloc(allocator, root);
    defer allocator.free(root_real);
    const full = try std.fs.path.join(allocator, &.{ root_real, rel });
    return full;
}

fn appendFileLineOriented(allocator: Allocator, path: []const u8, content: []const u8, unique: bool) !usize {
    const existing = std.fs.cwd().readFileAlloc(allocator, path, 64 * 1024 * 1024) catch |err| switch (err) {
        error.FileNotFound => try allocator.dupe(u8, ""),
        else => return err,
    };
    defer allocator.free(existing);
    var map = std.StringHashMap(void).init(allocator);
    defer map.deinit();
    if (unique) {
        var lines_existing = std.mem.splitScalar(u8, existing, '\n');
        while (lines_existing.next()) |line_raw| {
            const line = std.mem.trimRight(u8, line_raw, "\r");
            if (line.len > 0) try map.put(line, {});
        }
    }
    var out = std.ArrayList(u8).init(allocator);
    defer out.deinit();
    try out.appendSlice(existing);
    if (out.items.len > 0 and out.items[out.items.len - 1] != '\n') try out.append('\n');
    var appended: usize = 0;
    var lines_new = std.mem.splitScalar(u8, content, '\n');
    while (lines_new.next()) |line_raw| {
        const line = std.mem.trimRight(u8, line_raw, "\r");
        if (line.len == 0) continue;
        if (unique and map.contains(line)) continue;
        if (unique) {
            try map.put(line, {});
        }
        try out.appendSlice(line);
        try out.append('\n');
        appended += 1;
    }
    try atomicWriteFile(allocator, path, out.items);
    return appended;
}

fn replaceLines(allocator: Allocator, path: []const u8, start_line: usize, end_line: usize, content: []const u8) !void {
    const existing = try std.fs.cwd().readFileAlloc(allocator, path, 64 * 1024 * 1024);
    defer allocator.free(existing);
    var lines = std.ArrayList([]const u8).init(allocator);
    defer lines.deinit();
    var it = std.mem.splitScalar(u8, existing, '\n');
    while (it.next()) |line| try lines.append(std.mem.trimRight(u8, line, "\r"));
    if (existing.len > 0 and existing[existing.len - 1] == '\n' and lines.items.len > 0 and lines.items[lines.items.len - 1].len == 0) {
        _ = lines.pop();
    }
    if (start_line == 0 or end_line < start_line or start_line > lines.items.len + 1 or end_line > lines.items.len) return error.InvalidLineRange;
    var out = std.ArrayList(u8).init(allocator);
    defer out.deinit();
    var i: usize = 1;
    while (i < start_line and i <= lines.items.len) : (i += 1) {
        try out.appendSlice(lines.items[i - 1]);
        try out.append('\n');
    }
    var new_lines = std.mem.splitScalar(u8, content, '\n');
    while (new_lines.next()) |line_raw| {
        const line = std.mem.trimRight(u8, line_raw, "\r");
        try out.appendSlice(line);
        try out.append('\n');
    }
    i = end_line + 1;
    while (i <= lines.items.len) : (i += 1) {
        try out.appendSlice(lines.items[i - 1]);
        try out.append('\n');
    }
    try atomicWriteFile(allocator, path, out.items);
}

fn checkLines(allocator: Allocator, path: []const u8, lines_value: std.json.Value) ![]u8 {
    const existing = std.fs.cwd().readFileAlloc(allocator, path, 64 * 1024 * 1024) catch |err| switch (err) {
        error.FileNotFound => try allocator.dupe(u8, ""),
        else => return err,
    };
    defer allocator.free(existing);
    var map = std.StringHashMap(void).init(allocator);
    defer map.deinit();
    var existing_lines = std.mem.splitScalar(u8, existing, '\n');
    while (existing_lines.next()) |line_raw| {
        const line = std.mem.trimRight(u8, line_raw, "\r");
        try map.put(line, {});
    }
    var out = std.ArrayList(u8).init(allocator);
    const w = out.writer();
    try w.writeAll("[");
    switch (lines_value) {
        .array => |arr| {
            for (arr.items, 0..) |item, i| {
                if (i != 0) try w.writeAll(",");
                switch (item) {
                    .string => |s| try w.print("{}", .{map.contains(s)}),
                    else => try w.writeAll("false"),
                }
            }
        },
        else => {},
    }
    try w.writeAll("]");
    return out.toOwnedSlice();
}

fn atomicWriteFile(allocator: Allocator, path: []const u8, data: []const u8) !void {
    const parent = std.fs.path.dirname(path);
    if (parent) |p| try std.fs.cwd().makePath(p);
    const tmp = try std.fmt.allocPrint(allocator, "{s}.tmp.{}", .{ path, nowMillis() });
    defer allocator.free(tmp);
    {
        var file = try std.fs.cwd().createFile(tmp, .{ .truncate = true });
        defer file.close();
        try file.writeAll(data);
        try file.sync();
    }
    std.fs.cwd().rename(tmp, path) catch |err| {
        std.fs.cwd().deleteFile(tmp) catch {};
        return err;
    };
}

fn isHttpHostAllowed(allowed: []const u8, url: []const u8) bool {
    if (!std.mem.startsWith(u8, url, "https://") and !std.mem.startsWith(u8, url, "http://")) return false;
    if (std.mem.eql(u8, allowed, "*")) return true;
    const proto_end = std.mem.indexOf(u8, url, "://") orelse return false;
    const rest = url[proto_end + 3 ..];
    const host_end = std.mem.indexOfAny(u8, rest, "/:?#") orelse rest.len;
    const host = rest[0..host_end];
    if (host.len == 0) return false;
    var it = std.mem.splitScalar(u8, allowed, ',');
    while (it.next()) |item_raw| {
        const item = std.mem.trim(u8, item_raw, " \t\r\n");
        if (std.mem.eql(u8, item, "*") or std.ascii.eqlIgnoreCase(item, host)) return true;
    }
    return false;
}

fn joinUrl(allocator: Allocator, base: []const u8, suffix: []const u8) ![]u8 {
    if (std.mem.endsWith(u8, base, "/")) return std.fmt.allocPrint(allocator, "{s}{s}", .{ base, suffix });
    return std.fmt.allocPrint(allocator, "{s}/{s}", .{ base, suffix });
}


test "stableHash deterministic and seed sensitive" {
    const a = stableHash("autonomous agent runtime", 42);
    const b = stableHash("autonomous agent runtime", 42);
    const c = stableHash("autonomous agent runtime", 43);
    try std.testing.expectEqual(a, b);
    try std.testing.expect(a != c);
}

test "BitSet popcount intersection union and jaccard" {
    const gpa = std.testing.allocator;
    var set_a = try BitSet.init(gpa, 192);
    defer set_a.deinit();
    var set_b = try BitSet.init(gpa, 192);
    defer set_b.deinit();
    set_a.set(0);
    set_a.set(64);
    set_a.set(128);
    set_b.set(64);
    set_b.set(129);
    try std.testing.expectEqual(@as(usize, 1), set_a.intersectionPopCount(&set_b));
    try std.testing.expectEqual(@as(usize, 4), set_a.unionPopCount(&set_b));
    try std.testing.expect(set_a.get(64));
    try std.testing.expect(!set_a.get(65));
    const estimate = set_a.jaccardEstimate(&set_b);
    try std.testing.expectApproxEqAbs(@as(f32, 0.25), estimate, @as(f32, 0.001));
}

test "Tensor init deinit and shape" {
    const gpa = std.testing.allocator;
    var tensor = try Tensor.init(gpa, &.{ 3, 4 });
    defer tensor.deinit();
    try std.testing.expectEqual(@as(usize, 12), tensor.data.len);
    try std.testing.expectEqual(@as(usize, 2), tensor.shape.dims.len);
    try std.testing.expectEqual(@as(usize, 4), tensor.shape.dims[1]);
}

test "RankedSegment init and deinit" {
    const gpa = std.testing.allocator;
    var segment = try RankedSegment.init(gpa, &.{ 7, 8, 9 }, 0.5, 12, true);
    defer segment.deinit(gpa);
    try std.testing.expectEqual(@as(usize, 3), segment.tokens.len);
    try std.testing.expectEqual(@as(u64, 12), segment.position);
    try std.testing.expect(segment.anchor);
}

test "MGT dual language loads english and hungarian morphemes" {
    const gpa = std.testing.allocator;
    var mgt_dual = try MGT.init(gpa, &.{}, &.{}, null, .dual);
    defer mgt_dual.deinit();
    try std.testing.expect(mgt_dual.prefixes.contains("un"));
    try std.testing.expect(mgt_dual.prefixes.contains("meg"));
    try std.testing.expect(mgt_dual.suffixes.contains("ing"));
    try std.testing.expect(mgt_dual.suffixes.contains("ság"));
}

test "diffText LCS handles single line insertion" {
    const gpa = std.testing.allocator;
    const old_text = "alpha\nbeta\ngamma";
    const new_text = "alpha\ninserted\nbeta\ngamma";
    const diff = try diffText(gpa, old_text, new_text);
    defer gpa.free(diff);
    try std.testing.expect(std.mem.indexOf(u8, diff, "+2:inserted") != null);
    try std.testing.expect(std.mem.indexOf(u8, diff, "-") == null);
}

test "diffText LCS handles single line deletion" {
    const gpa = std.testing.allocator;
    const old_text = "alpha\nbeta\ngamma";
    const new_text = "alpha\ngamma";
    const diff = try diffText(gpa, old_text, new_text);
    defer gpa.free(diff);
    try std.testing.expect(std.mem.indexOf(u8, diff, "-2:beta") != null);
    try std.testing.expect(std.mem.indexOf(u8, diff, "+") == null);
}

test "diffText reports no changes" {
    const gpa = std.testing.allocator;
    const diff = try diffText(gpa, "same\ntext", "same\ntext");
    defer gpa.free(diff);
    try std.testing.expectEqualStrings("no changes\n", diff);
}

test "makeFtsQuery emits trailing token without delimiter" {
    const gpa = std.testing.allocator;
    const query = try makeFtsQuery(gpa, "planning status executing trailing");
    defer gpa.free(query);
    try std.testing.expect(std.mem.indexOf(u8, query, "trailing") != null);
    try std.testing.expect(std.mem.indexOf(u8, query, "planning") != null);
}

test "makeFtsQuery deduplicates tokens" {
    const gpa = std.testing.allocator;
    const query = try makeFtsQuery(gpa, "duplicate duplicate duplicate");
    defer gpa.free(query);
    const first = std.mem.indexOf(u8, query, "duplicate").?;
    const second = std.mem.indexOf(u8, query[first + 1 ..], "duplicate");
    try std.testing.expect(second == null);
}

test "appendFileLineOriented deduplicates within batch when unique" {
    const gpa = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir_path = try tmp.dir.realpathAlloc(gpa, ".");
    defer gpa.free(dir_path);
    const path = try std.fs.path.join(gpa, &.{ dir_path, "log.txt" });
    defer gpa.free(path);
    const appended = try appendFileLineOriented(gpa, path, "first\nfirst\nsecond\n", true);
    try std.testing.expectEqual(@as(usize, 2), appended);
    const data = try std.fs.cwd().readFileAlloc(gpa, path, 1024 * 1024);
    defer gpa.free(data);
    var count: usize = 0;
    var lines = std.mem.splitScalar(u8, data, '\n');
    while (lines.next()) |line| {
        if (std.mem.eql(u8, line, "first")) count += 1;
    }
    try std.testing.expectEqual(@as(usize, 1), count);
}

test "replaceLines rejects end line beyond file length" {
    const gpa = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir_path = try tmp.dir.realpathAlloc(gpa, ".");
    defer gpa.free(dir_path);
    const path = try std.fs.path.join(gpa, &.{ dir_path, "lines.txt" });
    defer gpa.free(path);
    try std.fs.cwd().writeFile(.{ .sub_path = path, .data = "one\ntwo\nthree\n" });
    const result = replaceLines(gpa, path, 1, 99, "replacement");
    try std.testing.expectError(error.InvalidLineRange, result);
    const unchanged = try std.fs.cwd().readFileAlloc(gpa, path, 1024 * 1024);
    defer gpa.free(unchanged);
    try std.testing.expectEqualStrings("one\ntwo\nthree\n", unchanged);
}

test "replaceLines swaps bounded range" {
    const gpa = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir_path = try tmp.dir.realpathAlloc(gpa, ".");
    defer gpa.free(dir_path);
    const path = try std.fs.path.join(gpa, &.{ dir_path, "lines2.txt" });
    defer gpa.free(path);
    try std.fs.cwd().writeFile(.{ .sub_path = path, .data = "one\ntwo\nthree\n" });
    try replaceLines(gpa, path, 2, 2, "replaced");
    const updated = try std.fs.cwd().readFileAlloc(gpa, path, 1024 * 1024);
    defer gpa.free(updated);
    try std.testing.expectEqualStrings("one\nreplaced\nthree\n", updated);
}

test "atomicWriteFile writes content and keeps destination on success" {
    const gpa = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir_path = try tmp.dir.realpathAlloc(gpa, ".");
    defer gpa.free(dir_path);
    const path = try std.fs.path.join(gpa, &.{ dir_path, "atomic.txt" });
    defer gpa.free(path);
    try atomicWriteFile(gpa, path, "payload");
    const data = try std.fs.cwd().readFileAlloc(gpa, path, 1024 * 1024);
    defer gpa.free(data);
    try std.testing.expectEqualStrings("payload", data);
}

test "evaluateStepVerifier passes intermediate step with done false state" {
    const gpa = std.testing.allocator;
    const state_json = "{\"done\":false,\"progress\":1}";
    const outcome_json = "{\"ok\":true,\"type\":\"compute\",\"result\":42}";
    const verifier = try evaluateStepVerifier(gpa, state_json, outcome_json);
    defer gpa.free(verifier);
    try std.testing.expect(std.mem.indexOf(u8, verifier, "\"success\":true") != null);
}

test "evaluateStepVerifier fails outcome reporting ok false" {
    const gpa = std.testing.allocator;
    const verifier = try evaluateStepVerifier(gpa, "{\"done\":false}", "{\"ok\":false,\"type\":\"tool_error\"}");
    defer gpa.free(verifier);
    try std.testing.expect(std.mem.indexOf(u8, verifier, "\"success\":false") != null);
}

test "validateValueBounded rejects both reasoning spellings" {
    const gpa = std.testing.allocator;
    const payload_space = "{\"note\":\"show your reasoning trace now\"}";
    try std.testing.expectError(error.ReasoningLeak, validateJsonText(gpa, payload_space));
    const payload_underscore = "{\"note\":\"show your reasoning_trace now\"}";
    try std.testing.expectError(error.ReasoningLeak, validateJsonText(gpa, payload_underscore));
}

test "CREV resolves conflict with confidence squaring" {
    const gpa = std.testing.allocator;
    var kernel = ChaosCoreKernel.init(gpa);
    defer kernel.deinit();
    var pipeline = try CREVPipeline.init(gpa, &kernel);
    defer pipeline.deinit();

    var incoming = try RelationalTriplet.init(gpa, "agent", "is_a", "planner", 0.9);
    defer incoming.deinit();
    var stronger = try RelationalTriplet.init(gpa, "agent", "is_not", "planner", 0.99);
    var conflicts = [_]*RelationalTriplet{&stronger};
    defer stronger.deinit();

    const resolved = try pipeline.resolveConflicts(&incoming, &conflicts);
    defer if (resolved != &incoming) {
        resolved.deinit();
        gpa.destroy(resolved);
    };
    try std.testing.expect(resolved != &incoming);
    const expected = (0.9 * 0.9 + 0.99 * 0.99) / (0.9 + 0.99);
    try std.testing.expectApproxEqAbs(expected, resolved.confidence, 0.0001);
}

test "CREV detects mutual exclusion contradictions" {
    const gpa = std.testing.allocator;
    var kernel = ChaosCoreKernel.init(gpa);
    defer kernel.deinit();
    var pipeline = try CREVPipeline.init(gpa, &kernel);
    defer pipeline.deinit();

    var has = try RelationalTriplet.init(gpa, "cache", "has", "entries", 0.9);
    defer has.deinit();
    var lacks = try RelationalTriplet.init(gpa, "cache", "lacks", "entries", 0.9);
    defer lacks.deinit();

    try std.testing.expect(!pipeline.checkConsistency(&has, &lacks));
    try std.testing.expect(pipeline.checkConsistency(&has, &has));
}

test "CREV full stream populates knowledge index and kernel memory" {
    const gpa = std.testing.allocator;
    var kernel = ChaosCoreKernel.init(gpa);
    defer kernel.deinit();
    var pipeline = try CREVPipeline.init(gpa, &kernel);
    defer pipeline.deinit();
    pipeline.setValidationThreshold(0.3);
    pipeline.tokenizer_config.min_confidence_threshold = 0.2;

    const result = try pipeline.processTextStream("The scheduler is a background service. The scheduler has retries.");
    try std.testing.expect(result.triplets_extracted >= 1);
    try std.testing.expect(pipeline.getKnowledgeGraphSize() >= 1);
    try std.testing.expect(kernel.memoryCount() >= 1);

    var matches = try pipeline.knowledge_index.queryMorphemeAware("the scheduler", null, null, gpa);
    defer matches.deinit();
    try std.testing.expect(matches.items.len >= 1);
}

test "ChaosCoreKernel allocates tagged memory and synchronizes graph" {
    const gpa = std.testing.allocator;
    var kernel = ChaosCoreKernel.init(gpa);
    defer kernel.deinit();
    const index = try kernel.allocateMemory("runtime|state|active|1.0", "triplet");
    try std.testing.expectEqual(@as(usize, 0), index);
    try std.testing.expectEqualStrings("runtime|state|active|1.0", kernel.getMemory(0).?);
    const synchronized = try kernel.synchronizeGraphWithMemory();
    try std.testing.expectEqual(@as(usize, 1), synchronized);
    try std.testing.expectEqual(@as(usize, 1), kernel.graph.nodeCount());
}

test "validatedFactsJson renders triplets as valid JSON" {
    const gpa = std.testing.allocator;
    var index = KnowledgeGraphIndex.init(gpa);
    defer index.deinit();
    const triplet = try gpa.create(RelationalTriplet);
    triplet.* = try RelationalTriplet.init(gpa, "agent", "owns", "workspace", 0.9);
    try index.index(triplet);
    const json = try validatedFactsJson(gpa, &index);
    defer gpa.free(json);
    var parsed = try std.json.parseFromSlice(std.json.Value, gpa, json, .{});
    defer parsed.deinit();
    const array = parsed.value.array;
    try std.testing.expectEqual(@as(usize, 1), array.items.len);
    try std.testing.expectEqualStrings("agent", objectGetString(array.items[0], "subject").?);
}

test "skillNumericId is deterministic per skill identifier" {
    const first = skillNumericId("skill_decompose_goal");
    const second = skillNumericId("skill_decompose_goal");
    const other = skillNumericId("skill_append_unique_log");
    try std.testing.expectEqual(first, second);
    try std.testing.expect(first != other);
}

test "MGT SSI Ranker retrieval chain ranks relevant skill first" {
    const gpa = std.testing.allocator;
    var mgt = try MGT.init(gpa, &.{}, &.{}, null, .dual);
    defer mgt.deinit();
    var ssi = SSI.init(gpa);
    defer ssi.deinit();
    var ranker = try Ranker.init(gpa, 4, 64, 0xA637200C4F12B551);
    defer ranker.deinit();

    var goal_tokens = std.ArrayList(u32).init(gpa);
    defer goal_tokens.deinit();
    try mgt.encode("split the goal into subgoals and record constraints", &goal_tokens);
    try ssi.addSequence(goal_tokens.items, skillNumericId("skill_goal"), true);

    var log_tokens = std.ArrayList(u32).init(gpa);
    defer log_tokens.deinit();
    try mgt.encode("append unique log lines without duplicates", &log_tokens);
    try ssi.addSequence(log_tokens.items, skillNumericId("skill_log"), true);

    var query_tokens = std.ArrayList(u32).init(gpa);
    defer query_tokens.deinit();
    try mgt.encode("the goal is planning and subgoals are empty", &query_tokens);

    const candidates = try ssi.retrieveTopK(query_tokens.items, 8, gpa);
    defer freeRankedSegments(candidates, gpa);
    try std.testing.expectEqual(@as(usize, 2), candidates.len);
    try ranker.rankCandidatesWithQuery(candidates, query_tokens.items, &ssi, gpa);
    try std.testing.expect(candidates[0].position == skillNumericId("skill_goal"));
}

test "tokensToJson produces parseable token array" {
    const gpa = std.testing.allocator;
    const tokens = [_]u32{ 10, 20, 30 };
    const json = try tokensToJson(gpa, &tokens);
    defer gpa.free(json);
    try std.testing.expectEqualStrings("[10,20,30]", json);
}

test "consumeBufferedLine shifts and shrinks buffer" {
    const gpa = std.testing.allocator;
    var buffer = std.ArrayList(u8).init(gpa);
    defer buffer.deinit();
    try buffer.appendSlice("first\nsecond\nthird");
    const nl = std.mem.indexOfScalar(u8, buffer.items, '\n').?;
    consumeBufferedLine(&buffer, nl);
    try std.testing.expectEqualStrings("second\nthird", buffer.items);
}

test "graph substrate nodes edges and coherence" {
    const gpa = std.testing.allocator;
    var graph = SelfSimilarRelationalGraph.init(gpa);
    defer graph.deinit();
    const node_a = try Node.initWithComplex(gpa, "n1", "subject", Complex(f64).init(0.8, 0.6), 0.1);
    try graph.addNode(node_a);
    const node_b = try Node.init(gpa, "n2", "object");
    try graph.addNode(node_b);
    var edge = try Edge.initWithComplex(gpa, "n1", "n2", .coherent, 0.9, Complex(f64).init(0.8, 0.6), 1.0);
    try edge.setMetadata("relation", "is_a");
    try graph.addEdge(edge);
    try std.testing.expectEqual(@as(usize, 2), graph.nodeCount());
    try std.testing.expectEqual(@as(usize, 1), graph.edgeCount());
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), graph.coherenceRatio(), 0.0001);
    try std.testing.expectEqualStrings("is_a", graph.edges.items[0].getMetadata("relation").?);
}

test "extractJsonObject finds embedded envelope" {
    const gpa = std.testing.allocator;
    const text = "prefix noise {\"state_patch\":{\"done\":true},\"action\":{\"type\":\"finish\",\"args\":{}}} trailing";
    const extracted = try extractJsonObject(gpa, text);
    defer gpa.free(extracted);
    var parsed = try std.json.parseFromSlice(std.json.Value, gpa, extracted, .{});
    defer parsed.deinit();
    try std.testing.expect(objectGetValue(parsed.value, "state_patch") != null);
}

test "isHttpHostAllowed enforces scheme and host list" {
    try std.testing.expect(isHttpHostAllowed("*", "https://example.com/api"));
    try std.testing.expect(isHttpHostAllowed("example.com,api.local", "http://api.local/x"));
    try std.testing.expect(!isHttpHostAllowed("example.com", "http://api.local/x"));
    try std.testing.expect(!isHttpHostAllowed("*", "ftp://example.com/x"));
}


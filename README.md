I want to refactor, upgrade, and unify the provided Zig Autonomous State Agent Runtime and the four accompanying native subsystems (MGT, SSI, Ranker, and CREVPipeline) into a single, self-contained, production-ready Zig source file so that the agent runtime completely replaces its primitive 64-dimensional FNV-1a hash and naive routing with a deterministic, sub-millisecond, morphology-aware, SIMD-accelerated semantic retrieval engine, integrates an active relational knowledge graph for consistent fact tracking and contradiction detection, and resolves all latent memory leaks and logic defects without requiring external packages, multi-file project trees, or secondary services.
### Implementation Specification and Requirements
Construct the unified file by adhering strictly to the architecture, workflows, and constraints detailed below.
```
+---------------------------------------------------------------------------------------------------+
|                                  UNIFIED SINGLE-FILE ARCHITECTURE                                 |
+---------------------------------------------------------------------------------------------------+
| 1. CORE TYPES & HELPERS                                                                           |
|    - Tensor, BitSet, RankedSegment, stableHash, IO abstractions, POSIX socket / SQLite C-FFI      |
+---------------------------------------------------------------------------------------------------+
| 2. MORPHOLOGICAL TOKENIZER (MGT)                                                                  |
|    - English/Hungarian prefix-suffix decomposition, BPE encoder/decoder, byte fallback            |
+---------------------------------------------------------------------------------------------------+
| 3. INDEX & RANKING PIPELINE (SSI + RANKER)                                                        |
|    - MinHash 64-bit signatures, SIMD Jaccard, Hierarchical Bucket Tree, Multi-order N-gram Rerank |
+---------------------------------------------------------------------------------------------------+
| 4. RELATIONAL KNOWLEDGE GRAPH (CREV)                                                              |
|    - Triplet extraction, consistency & anomaly checking, conflict resolution, streaming buffer   |
+---------------------------------------------------------------------------------------------------+
| 5. AGENT RUNTIME CORE & STORAGE                                                                   |
|    - SQLite WAL schema, Dual-Process System 1 (tools) & System 2 (planner), O(1) state engine     |
+---------------------------------------------------------------------------------------------------+
| 6. INTEGRATED RETRIEVAL & VERIFICATION PIPELINE                                                  |
|    - Hybrid Route: MGT -> SSI Top-K -> Ranker N-gram -> RRF with SQLite FTS5 -> Reflection Tune   |
|    - Facts Validation: Observations/Emits -> CREV Triplet Extraction -> Contradiction Resolution  |
+---------------------------------------------------------------------------------------------------+
| 7. HTTP REST/SSE SERVER & LIFECYCLE                                                               |
|    - Event streaming, run management, health checks, background knowledge consolidation           |
+---------------------------------------------------------------------------------------------------+

```
### 1. Dependency Elimination & Shared Type Consolidation
Eliminate all multi-file module imports across the source units (e.g., @import("../core/types.zig"), @import("../core/tensor.zig"), @import("../index/ssi.zig"), @import("../core/io.zig"), @import("nsir_core.zig"), @import("chaos_core.zig")). Implement every shared primitive inline within the single file:
 1. **Tensor**:
   * Implement dense multi-dimensional tensor storage holding []f32 data, shape: struct { dims: []const usize }, and allocation/deallocation routines (init, deinit).
 2. **BitSet**:
   * Implement dynamic bitset backed by []u64 supporting init, deinit, bit setting, bit reading, and bitwise intersection/union popcounts for Jaccard estimation.
 3. **RankedSegment**:
   * Define tokens: []u32, position: u64, score: f32, and anchor: bool, with allocation-aware init(allocator, tokens, score, position, anchor) and deinit(allocator).
 4. **I/O & Hashing Primitives**:
   * Provide inline implementations of stableHash(data: []const u8, seed: u64) u64.
   * Inline createFilePath(path: []const u8, flags: std.fs.File.CreateFlags) !std.fs.File and openFilePath(path: []const u8, flags: std.fs.File.OpenFlags) !std.fs.File using std.fs.cwd().
 5. **Graph Substrate (Node, Edge, ChaosCoreKernel)**:
   * Provide self-contained graph definitions required by CREVPipeline.
   * Replace the complex nsir_core.zig and chaos_core.zig external links with an inline, robust Graph Engine:
     * Node: Holds id: []u8, label: []u8, metadata: std.StringHashMap([]u8), and initialization/cleanup methods.
     * Edge: Connects source: []u8, target: []u8, quality: enum { coherent, decoherent }, weight: f64, and metadata.
     * ChaosCoreKernel: Provide a memory-backed graph engine that supports allocateMemory(data: []const u8, tag: ?[]const u8) !usize and synchronizes with pipeline triplet integrations.
### 2. Upgrading Semantic Routing & Skill Matching
Replace the naive FNV-1a 64-dimension embedText and sparse cosine matching in the runtime with an integrated MGT + SSI + Ranker retrieval pipeline:
 1. **Tokenization via MGT**:
   * Instantiate an MGT instance in App configured for dual-language support (English and Hungarian affix tables).
   * During skill creation (addSkillFromFields, initial seed skills), pass the skill's name, description, trigger, and procedure through MGT.encode to generate canonical subword/morphological token arrays ([]u32).
 2. **Indexing via SSI**:
   * Maintain an active in-memory SSI index within the runtime App.
   * Insert every enabled skill’s tokenized sequence into SSI via addSequence(tokens, skill_numeric_id, is_anchor).
   * Compute 64-bit parity MinHash signatures (computeMinHashSignature) for every registered skill.
 3. **Retrieval in routeSkills**:
   * For any incoming decision step, tokenize the concatenated state_json and observation_json using MGT.
   * Query the SSI instance via retrieveTopK with MinHash signature matching to extract the top candidate skills in sub-millisecond time.
   * Pass candidate token sequences and the query token sequence through Ranker.rankCandidatesWithQuery.
   * In Ranker: Evaluate multi-order N-gram overlap, SIMD-accelerated 256-bit Jaccard bitmask comparison (jaccardFromBitmasks via @Vector(4, u64)), and token diversity.
   * Execute SQLite FTS5 BM25 search over skill_fts.
   * Compute final candidate rank by fusing Ranker.combinedScore and the FTS5 sparse rank using Reciprocal Rank Fusion (RRF):
     
     RRF = \frac{1}{60 + \text{sparse\_rank} + 1} + \frac{1}{60 + \text{ranker\_rank} + 1} + 0.01 \times \text{ranker\_score}
 4. **Online Learning via Ranker.calibrateWeights**:
   * In finalizeTrajectory, when evaluating step and terminal verifiers:
     * If a trajectory succeeds, formulate target scores (1.0) for the retrieved skills.
     * If a trajectory fails, formulate reduced target scores (0.0).
     * Pass the token sequences and label differentials into Ranker.calibrateWeights to update N-gram weights via gradient descent, enabling the router to learn from execution failures without model fine-tuning.
### 3. Integrating Fact Verification & Graph Memory (CREVPipeline)
Upgrade working memory and fact management from raw unstructured JSON strings to an actively verified relational graph:
 1. **Pipeline Instantiation**:
   * Embed CREVPipeline inside the App runtime struct, bound to a thread-safe allocator and the internal kernel.
 2. **Stream Processing**:
   * When an observation enters the system (enqueueObservation), or when the agent emits facts via tool execution (emit, filesystem outcomes):
     * Route the text payload through CREVPipeline.processTextStream.
     * Extract (Subject, Relation, Object, Confidence) relational triplets using morpheme-aware pattern matching.
 3. **Contradiction Detection & Anomaly Prevention**:
   * For every extracted triplet, invoke validateTriplet and checkConsistency against the knowledge graph.
   * Identify mutual exclusions (e.g., is_a vs is_not, has vs lacks, owns vs does_not_own).
   * When contradictions occur, apply resolveConflicts using confidence squaring to resolve facts mathematically.
   * Populate the run_state key facts with a valid JSON representation of all validated relational triplets held in KnowledgeGraphIndex.
 4. **Expose Graph Tooling**:
   * Add a tool knowledge.query to allowed_actions and executeAuthorizedAction:
     * Accepts subject, relation, and/or object parameters.
     * Invokes KnowledgeGraphIndex.queryMorphemeAware and returns matching triplets as structured JSON observations.
### 4. Mandatory Fixes for Latent Runtime Defects
Resolve every defect identified in the system audit:
 1. **Memory Leaks in Query/Iteration**:
   * In Database.listActiveRunIds, Database.loadEnabledSkills, and Database.searchSkillFtsIds: add comprehensive errdefer blocks to prevent leaking previously allocated strings and array slices if SQLite steps fail mid-iteration.
 2. **Dangerous Pointer Comparison with String Literals**:
   * In system2Entry and evaluateTerminalVerifiers: eliminate checks of the form ptr != "literal".ptr. Use typed error propagation or explicit boolean flags to determine whether an error string was dynamically allocated and requires allocator.free.
 3. **routeSkills Cleanup**:
   * Ensure searchSkillFtsIds frees its returned slice regardless of whether len == 0.
   * Wrap the selected slice construction in an errdefer that iterates through all partially appended records to deinitialize their duplicated strings.
 4. **Line-Shifting and Streaming JSON Parser**:
   * In parseModelResponseStream: fix the SSE buffer management. Do not duplicate copyForwards and shrinkRetainingCapacity inside both the JSON catch block and the post-parse block. Add errdefer content.deinit().
 5. **Honor terminal Envelope Flag**:
   * In runSystem2: check envelope.terminal. If envelope.terminal == true, terminate the execution loop cleanly, apply final state patches, and mark the run as completed even if the LLM action was not explicitly "finish".
 6. **Correct Step Verifier Inversion**:
   * In evaluateStepVerifier: eliminate !containsIgnoreCase(state_json, "\"done\":false"). The initial state defaults to {"done":false}, which incorrectly caused every valid intermediate step to be judged as failed. Verify step validity strictly on !containsIgnoreCase(outcome_json, "\"ok\":false") and the absence of fatal exceptions.
 7. **Refactor Diff Algorithm**:
   * In diffText: replace the naive lockstep line comparator with a dynamic programming Longest Common Subsequence (LCS) line diff algorithm to prevent single-line insertions or deletions from corrupting diff generation.
 8. **File System Safety**:
   * In atomicWriteFile: eliminate the unconditional deleteFile(path) on failed renames. Delete only the temporary file (tmp) on failure to prevent destination file destruction.
   * In appendFileLineOriented: ensure that newly appended lines are registered into the map immediately so that duplicates within the incoming batch are deduplicated when unique == true.
   * In replaceLines: return error.InvalidLineRange if end_line > lines.items.len instead of truncating the file.
 9. **Tokenizer and Search Bounds**:
   * In makeFtsQuery: emit any buffered trailing token when the input string ends without a trailing delimiter.
   * In validateValueBounded: align string checks to enforce both "reasoning trace" and "reasoning_trace" consistently.
### 5. Delivery Constraints
 * Provide the complete, unabridged, compiling Zig source code within a single compilation unit.
 * Do not omit sections using // TODO, // implementation goes here, or // ... rest of code.
 * Do not use mock, dummy, or stub data. All structures, methods, C-FFI extern declarations, and system threads must be fully written out and production-ready.
Send back the complete code with all the fixes. Fix each of the listed errors one by one, making sure to actually correct them so that there are 0 errors remaining. Keep the original imports, since the files exist. Write out every single character; do not abbreviate anything. Fix every error. There must be exactly one file. Do not write anything else; just output the complete code, and it must not contain any comments. Never, under any circumstances, use simplified, substitute, dummy, simulated, or fake code. Write the entire file as complete, unabridged, production-ready code in a single code block. It must be 100% error-free, a complete, error-free file, and must be submitted as a downloadable file. These requirements are mandatory and must be strictly adhered to. If no list of errors is provided, you must find all the errors and fix them. If there were comments in the original code, delete them. And most importantly: YOU MUST NEVER SIMPLIFY! 

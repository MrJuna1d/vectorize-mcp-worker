import { describe, it, expect } from 'vitest';
import { reflectionEngine } from '../src/engines/reflection';
import type { Document } from '../src/types/document';

// ---------------------------------------------------------------------------
// A richer mock than the call-counting one — this one RECORDS the actual SQL
// binds and Vectorize upsert payloads so we can assert on real content, not
// just "was it called".
// ---------------------------------------------------------------------------
function makeRecordingEnv(opts: {
	matches: { id: string; score: number; content: string; doc_type?: string }[];
	queryByIdShouldFail?: boolean;
}) {
	const dbInserts: { sql: string; args: any[] }[] = [];
	const dbUpdates: { sql: string; args: any[] }[] = [];
	const vectorizeUpserts: any[] = [];

	const env: any = {
		AI: {
			run: async (_model: string, input: any) => {
				if (input?.messages) {
					return { response: 'INSIGHT: adds new detail. CONNECTION: links to prior doc. GAP: timeline unclear.' };
				}
				return { data: [new Array(8).fill(0.1)] }; // small fake embedding
			},
		},
		VECTORIZE: {
			query: async () => ({
				matches: opts.matches.map(m => ({
					id: m.id,
					score: m.score,
					metadata: { content: m.content, doc_type: m.doc_type ?? 'raw' },
				})),
			}),
			queryById: async () => {
				if (opts.queryByIdShouldFail) throw new Error('not yet queryable');
				return {
					matches: opts.matches.map(m => ({
						id: m.id,
						score: m.score,
						metadata: { content: m.content, doc_type: m.doc_type ?? 'raw' },
					})),
				};
			},
			upsert: async (vecs: any[]) => {
				vectorizeUpserts.push(...vecs);
				return {};
			},
		},
		DB: {
			prepare: (sql: string) => ({
				bind: (...args: any[]) => ({
					run: async () => {
						if (/^\s*INSERT/i.test(sql)) dbInserts.push({ sql, args });
						if (/^\s*UPDATE/i.test(sql)) dbUpdates.push({ sql, args });
						return {};
					},
					first: async () => null,
					all: async () => ({ results: [] }),
				}),
			}),
		},
		EMBEDDING_MODEL: 'qwen3-0.6b',
		REFLECTION_MODEL: 'kimi-k2.5',
	};

	return { env, dbInserts, dbUpdates, vectorizeUpserts };
}

function doc(id: string, content = 'A new document about something specific.'): Document {
	return { id, content };
}

describe('reflect() correctness — verifies WHAT gets written, not just whether something ran', () => {
	it('writes a D1 reflection row with correct doc_type, parent_ids, and reflection_score', async () => {
		const { env, dbInserts } = makeRecordingEnv({
			matches: [
				{ id: 'other-1-chunk-0', score: 0.8, content: 'Related content A' },
				{ id: 'other-2-chunk-0', score: 0.7, content: 'Related content B' },
			],
		});

		await reflectionEngine.reflect(doc('new-doc'), env, 'new-doc-chunk-0');

		const reflectionInsert = dbInserts.find(i => /INTO documents/i.test(i.sql) && /doc_type/i.test(i.sql));
		expect(reflectionInsert).toBeDefined();

		// bind order from reflection.ts:
		// (id, content, title, source, category, chunk_index, parent_id,
		//  word_count, is_image, tenant_id, doc_type, parent_ids, last_reflected_at, reflection_version)
		const args = reflectionInsert!.args;
		const [id, content, , , , , parentId, , , , docType, parentIdsJson] = args;

		expect(id).toMatch(/^reflection_new-doc_\d+$/);
		expect(content).toContain('INSIGHT');
		expect(parentId).toBe('new-doc');
		expect(docType).toBe('reflection');

		const parentIds = JSON.parse(parentIdsJson);
		expect(parentIds).toContain('new-doc');
		expect(parentIds).toContain('other-1-chunk-0');
		expect(parentIds).toContain('other-2-chunk-0');
	});

	it('computes reflection_score as the mean of the related match scores', async () => {
		const { env, vectorizeUpserts } = makeRecordingEnv({
			matches: [
				{ id: 'a-chunk-0', score: 0.9, content: 'A' },
				{ id: 'b-chunk-0', score: 0.7, content: 'B' },
			],
		});

		await reflectionEngine.reflect(doc('score-doc'), env, 'score-doc-chunk-0');

		expect(vectorizeUpserts.length).toBe(1);
		const reflectionScore = vectorizeUpserts[0].metadata.reflection_score;
		expect(reflectionScore).toBeCloseTo((0.9 + 0.7) / 2, 5);
	});

	it('stamps last_reflected_at on the source document via UPDATE', async () => {
		const { env, dbUpdates } = makeRecordingEnv({
			matches: [{ id: 'x-chunk-0', score: 0.8, content: 'X' }],
		});

		await reflectionEngine.reflect(doc('stamped-doc'), env, 'stamped-doc-chunk-0');

		const stampUpdate = dbUpdates.find(u => /last_reflected_at/i.test(u.sql));
		expect(stampUpdate).toBeDefined();
		expect(stampUpdate!.args).toContain('stamped-doc');
	});

	it('Vectorize metadata matches the D1 row — same score, same content, same doc_type', async () => {
		const { env, dbInserts, vectorizeUpserts } = makeRecordingEnv({
			matches: [{ id: 'y-chunk-0', score: 0.75, content: 'Y content' }],
		});

		await reflectionEngine.reflect(doc('sync-doc'), env, 'sync-doc-chunk-0');

		const reflectionInsert = dbInserts.find(i => /doc_type/i.test(i.sql));
		const [, d1Content] = reflectionInsert!.args;

		const vec = vectorizeUpserts[0];
		expect(vec.metadata.content).toBe(d1Content);
		expect(vec.metadata.doc_type).toBe('reflection');
		expect(vec.metadata.reflection_score).toBeCloseTo(0.75, 5);
	});

	it('optimized path (queryById hit) produces IDENTICAL written content to the fallback path', async () => {
		const matches = [{ id: 'z-chunk-0', score: 0.85, content: 'Z content' }];

		const optimized = makeRecordingEnv({ matches });
		await reflectionEngine.reflect(doc('same-doc', 'Same content'), optimized.env, 'same-doc-chunk-0');

		const fallback = makeRecordingEnv({ matches, queryByIdShouldFail: true });
		await reflectionEngine.reflect(doc('same-doc', 'Same content'), fallback.env, 'same-doc-chunk-0');

		// Both paths must agree on the parts that don't depend on Date.now()/random LLM text
		const optInsert = optimized.dbInserts.find(i => /doc_type/i.test(i.sql))!;
		const fbInsert = fallback.dbInserts.find(i => /doc_type/i.test(i.sql))!;

		expect(optInsert.args[6]).toBe(fbInsert.args[6]); // parent_id
		expect(optInsert.args[10]).toBe(fbInsert.args[10]); // doc_type
		expect(JSON.parse(optInsert.args[11])).toEqual(JSON.parse(fbInsert.args[11])); // parent_ids

		expect(optimized.vectorizeUpserts[0].metadata.reflection_score)
			.toBeCloseTo(fallback.vectorizeUpserts[0].metadata.reflection_score, 5);
	});

	it('writes NOTHING when no related chunks clear the similarity threshold', async () => {
		const { env, dbInserts, dbUpdates, vectorizeUpserts } = makeRecordingEnv({
			matches: [{ id: 'weak-chunk-0', score: 0.1, content: 'Barely related' }],
		});

		await reflectionEngine.reflect(doc('lonely-doc'), env, 'lonely-doc-chunk-0');

		expect(dbInserts.length).toBe(0);
		expect(dbUpdates.length).toBe(0);
		expect(vectorizeUpserts.length).toBe(0);
	});

	it('excludes the source document\'s own chunks and existing reflections/summaries from parent_ids', async () => {
		const { env, dbInserts } = makeRecordingEnv({
			matches: [
				{ id: 'exclude-doc-chunk-1', score: 0.99, content: 'self chunk' }, // same doc, should be excluded
				{ id: 'old-reflection-1', score: 0.95, content: 'stale reflection', doc_type: 'reflection' }, // should be excluded
				{ id: 'genuine-related-chunk-0', score: 0.8, content: 'real related content' }, // should be kept
			],
		});

		await reflectionEngine.reflect(doc('exclude-doc'), env, 'exclude-doc-chunk-0');

		const reflectionInsert = dbInserts.find(i => /doc_type/i.test(i.sql));
		expect(reflectionInsert).toBeDefined();
		const parentIds = JSON.parse(reflectionInsert!.args[11]);

		expect(parentIds).toContain('genuine-related-chunk-0');
		expect(parentIds).not.toContain('exclude-doc-chunk-1');
		expect(parentIds).not.toContain('old-reflection-1');
	});
});
import * as ort from 'onnxruntime-web';

ort.env.wasm.numThreads = 1;

function resolveAssetUrl(filename: string): string {
  return new URL(filename, window.location.origin + import.meta.env.BASE_URL).toString();
}

function buildByteCodec(): { encoder: Map<number, string>; decoder: Map<string, number> } {
  const bs: number[] = [];
  const cs: number[] = [];

  for (let i = 33; i <= 126; i++) {
    bs.push(i);
    cs.push(i);
  }

  for (let i = 161; i <= 172; i++) {
    bs.push(i);
    cs.push(i);
  }

  for (let i = 174; i <= 255; i++) {
    bs.push(i);
    cs.push(i);
  }

  const printable = new Set(bs);

  let n = 0;

  for (let b = 0; b < 256; b++) {
    if (!printable.has(b)) {
      bs.push(b);
      cs.push(256 + n);
      n++;
    }
  }

  const encoder = new Map<number, string>();
  const decoder = new Map<string, number>();

  for (let i = 0; i < bs.length; i++) {
    const ch = String.fromCodePoint(cs[i]);
    encoder.set(bs[i], ch);
    decoder.set(ch, bs[i]);
  }

  return { encoder, decoder };
}

const BYTE_CODEC = buildByteCodec();

const PRE_TOKENIZE_RE = /'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;

function applyBPE(token: string, mergeRanks: Map<string, number>): string[] {
  let word = Array.from(token);

  if (word.length <= 1) {
    return word;
  }

  while (word.length > 1) {
    let bestRank = Infinity;
    let bestLeft = '';
    let bestRight = '';

    for (let i = 0; i < word.length - 1; i++) {
      const rank = mergeRanks.get(`${word[i]} ${word[i + 1]}`);

      if (rank !== undefined && rank < bestRank) {
        bestRank = rank;
        bestLeft = word[i];
        bestRight = word[i + 1];
      }
    }

    if (bestRank === Infinity) break;

    const merged = bestLeft + bestRight;
    const next: string[] = [];

    let i = 0;

    while (i < word.length) {
      if (i < word.length - 1 && word[i] === bestLeft && word[i + 1] === bestRight) {
        next.push(merged);
        i += 2;
      } else {
        next.push(word[i]);
        i++;
      }
    }

    word = next;
  }

  return word;
}

function bpeEncode(text: string, vocab: Record<string, number>, mergeRanks: Map<string, number>): number[] {
  const words = text.match(PRE_TOKENIZE_RE) || [];
  const ids: number[] = [];
  const utf8 = new TextEncoder();

  for (const word of words) {
    const byteChars = Array.from(utf8.encode(word))
      .map((b) => BYTE_CODEC.encoder.get(b)!)
      .join('');

    const subwords = applyBPE(byteChars, mergeRanks);

    for (const sw of subwords) {
      const id = vocab[sw];
      ids.push(id !== undefined ? id : 0);
    }
  }

  return ids;
}

function bpeDecode(ids: number[], idToToken: Record<number, string>): string {
  const raw: number[] = [];

  for (const id of ids) {
    const tok = idToToken[id];

    if (!tok) {
      continue;
    }

    if (tok.startsWith('<|') && tok.endsWith('|>')) {
      continue;
    }

    for (const ch of tok) {
      const b = BYTE_CODEC.decoder.get(ch);
      if (b !== undefined) raw.push(b);
    }
  }

  return new TextDecoder().decode(new Uint8Array(raw));
}

export interface ModelState {
  session: ort.InferenceSession;
  vocab: Record<string, number>;
  idToToken: Record<number, string>;
  specialTokens: Record<string, number>;
  mergeRanks: Map<string, number>;
}

export async function loadModel(onStatus: (msg: string) => void): Promise<ModelState> {
  onStatus('Loading tokenizer…');

  const tokenizerUrl = resolveAssetUrl('tokenizer.json');
  const response = await fetch(tokenizerUrl);

  if (!response.ok) {
    throw new Error(`Failed to load tokenizer.json (${response.status}) from ${tokenizerUrl}`);
  }

  const responseText = await response.text();
  let vocabData: any;

  try {
    vocabData = JSON.parse(responseText);
  } catch {
    const preview = responseText.slice(0, 80).replace(/\s+/g, ' ');
    throw new Error(`tokenizer.json at ${tokenizerUrl} is not valid JSON (starts with: "${preview}")`);
  }

  onStatus('Loading ONNX model…');
  const modelUrl = resolveAssetUrl('picorouter.onnx');

  const session = await ort.InferenceSession.create(modelUrl, {
    executionProviders: ['wasm'],
  });

  const vocab: Record<string, number> = vocabData.model?.vocab || {};
  const idToToken: Record<number, string> = {};

  for (const [token, id] of Object.entries(vocab)) {
    idToToken[id as number] = token;
  }

  const specialTokens: Record<string, number> = {};

  if (vocabData.added_tokens) {
    for (const t of vocabData.added_tokens) {
      specialTokens[t.content] = t.id;
      idToToken[t.id] = t.content;
    }
  }

  const mergeRanks = new Map<string, number>();
  const rawMerges: [string, string][] = vocabData.model?.merges || [];

  for (let i = 0; i < rawMerges.length; i++) {
    mergeRanks.set(`${rawMerges[i][0]} ${rawMerges[i][1]}`, i);
  }

  const vocabSize = Object.keys(vocab).length;
  onStatus(`Model loaded (${vocabSize} vocab, ${rawMerges.length} merges)`);

  return { session, vocab, idToToken, specialTokens, mergeRanks };
}

export async function runInference(
  model: ModelState,
  context: string,
  question: string,
  onStatus: (msg: string) => void,
): Promise<string> {
  const { session, vocab, idToToken, specialTokens, mergeRanks } = model;

  const contextIds = bpeEncode(context, vocab, mergeRanks);
  const questionIds = bpeEncode(question, vocab, mergeRanks);

  const inputIds = [
    specialTokens['<|context|>'] ?? 2,
    ...contextIds,
    specialTokens['<|tools|>'] ?? 3,
    specialTokens['<|user|>'] ?? 4,
    ...questionIds,
    specialTokens['<|assistant|>'] ?? 5,
  ];

  onStatus(`Input: ${inputIds.length} tokens`);

  const generatedTokens: number[] = [];
  const currentIds = [...inputIds];
  const maxNewTokens = 50;
  const eosId = specialTokens['<|eos|>'] ?? 1;

  for (let i = 0; i < maxNewTokens; i++) {
    const tensor = new ort.Tensor('int64', BigInt64Array.from(currentIds.map(BigInt)), [1, currentIds.length]);

    const out = await session.run({ input_ids: tensor });
    const logitsData = out.logits.data as Float32Array;

    const vocabSize = out.logits.dims[2];
    const offset = (currentIds.length - 1) * vocabSize;

    let maxIdx = 0;
    let maxVal = logitsData[offset];

    for (let j = 1; j < vocabSize; j++) {
      if (logitsData[offset + j] > maxVal) {
        maxVal = logitsData[offset + j];
        maxIdx = j;
      }
    }

    if (maxIdx === eosId) {
      break;
    }

    generatedTokens.push(maxIdx);
    currentIds.push(maxIdx);
  }

  const decoded = bpeDecode(generatedTokens, idToToken);

  onStatus(`Inference complete — ${generatedTokens.length} tokens generated`);

  return `Generated (${generatedTokens.length} tokens):\n${decoded}\n\nToken IDs: [${generatedTokens.join(', ')}]`;
}

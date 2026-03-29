import * as ort from 'onnxruntime-web';

ort.env.wasm.numThreads = 1;

export interface ModelState {
  session: ort.InferenceSession;
  vocab: Record<string, number>;
  idToToken: Record<number, string>;
  specialTokens: Record<string, number>;
}

export async function loadModel(onStatus: (msg: string) => void): Promise<ModelState> {
  onStatus('Loading tokenizer…');

  const response = await fetch('./tokenizer.json');

  if (!response.ok) {
    throw new Error(`Failed to load tokenizer.json: ${response.status}`);
  }

  const vocabData = await response.json();

  onStatus('Loading ONNX model…');

  const session = await ort.InferenceSession.create('./picorouter.onnx', {
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

  const vocabSize = Object.keys(vocab).length;

  onStatus(`Model loaded (${vocabSize} vocab tokens)`);

  return { session, vocab, idToToken, specialTokens };
}

export async function runInference(
  model: ModelState,
  context: string,
  question: string,
  onStatus: (msg: string) => void,
): Promise<string> {
  const { session, vocab, idToToken, specialTokens } = model;

  const contextIds = simpleEncode(context, vocab);
  const questionIds = simpleEncode(question, vocab);

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

  const decoded = simpleDecode(generatedTokens, idToToken);

  onStatus(`Inference complete — ${generatedTokens.length} tokens generated`);

  return `Generated (${generatedTokens.length} tokens):\n${decoded}\n\nToken IDs: [${generatedTokens.join(', ')}]`;
}

function simpleEncode(text: string, vocab: Record<string, number>): number[] {
  const tokens = text.toLowerCase().match(/\w+|[^\s\w]/g) || [];
  return tokens.map((t) => vocab[t] ?? vocab[`Ġ${t}`] ?? vocab['<|pad|>'] ?? 0);
}

function simpleDecode(ids: number[], idToToken: Record<number, string>): string {
  return ids
    .map((id) => {
      const tok = idToToken[id] || `[${id}]`;
      return tok.replace(/^Ġ/, ' ');
    })
    .join('');
}

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { loadModel, runInference, type ModelState } from './inference';
import { search, splitIntoPassages } from './retrieval';

const DEFAULT_CONTEXT = `
The Eiffel Tower was completed in 1889 for the World's Fair in Paris.
It was designed by engineer Gustave Eiffel and stands 330 meters tall.
The tower is located on the Champ de Mars near the Seine River and
attracts approximately 7 million visitors each year.
`.trim();

export default function App() {
  const [context, setContext] = useState(DEFAULT_CONTEXT);
  const [question, setQuestion] = useState('When was the Eiffel Tower completed?');
  const [output, setOutput] = useState('Loading model…');
  const [status, setStatus] = useState('');
  const [loading, setLoading] = useState(true);
  const [running, setRunning] = useState(false);

  const modelRef = useRef<ModelState | null>(null);
  const passages = useMemo(() => splitIntoPassages(context), [context]);
  const retrievalResults = useMemo(() => search(question, passages, { topK: 3 }), [question, passages]);

  useEffect(() => {
    loadModel(setStatus)
      .then((state) => {
        modelRef.current = state;
        setOutput('Model loaded. Type a question and click Ask.');
        setLoading(false);
      })
      .catch((error: any) => {
        console.error(error);

        setOutput(`Error loading model: Make sure picorouter.onnx and tokenizer.json are in public/.`);

        setStatus('Load failed');
        setLoading(false);
      });
  }, []);

  const handleAsk = useCallback(async () => {
    const model = modelRef.current;

    if (!model) {
      return;
    }

    setRunning(true);

    setOutput('Running inference…');

    try {
      const retrievedContext = retrievalResults.map((result, i) => `[${i + 1}] ${result.document}`).join('\n\n');

      const finalContext = retrievedContext || context;

      setStatus(
        `Retrieved ${retrievalResults.length || 1} passage(s) from ${passages.length || 1} total, sending to model…`,
      );

      const result = await runInference(model, finalContext, question, setStatus);
      const retrievalPreview = retrievedContext || '[1] No retrieval match score > 0; using raw context.';

      setOutput(`${result}\n\nRetrieved passages:\n${retrievalPreview}`);
    } catch (error) {
      setOutput(`Inference error: ${error instanceof Error ? error.message : error}`);
    } finally {
      setRunning(false);
    }
  }, [context, passages.length, question, retrievalResults]);

  const disabled = loading || running;

  return (
    <div className="mx-auto max-w-2xl px-4 py-8">
      <h1 className="text-2xl font-bold">PicoRouter</h1>
      <p className="mt-1 mb-6 text-sm text-gray-500">Tracer bullet — end-to-end pipeline test</p>

      <div className="mb-4 rounded-lg border border-gray-200 bg-white p-4">
        <label className="mb-2 block text-xs font-semibold uppercase tracking-wide text-gray-400">Context corpus</label>
        <textarea
          className="w-full resize-y rounded border border-gray-200 p-2 text-sm leading-relaxed focus:border-gray-400 focus:outline-none"
          rows={4}
          value={context}
          onChange={(event) => setContext(event.target.value)}
        />
        <p className="mt-2 text-xs text-gray-400">
          Retrieval splits this into {passages.length} passage{passages.length === 1 ? '' : 's'} and sends top matches
          to the model.
        </p>
      </div>

      <div className="mb-4 flex gap-2">
        <input
          type="text"
          className="flex-1 rounded-md border border-gray-200 px-3 py-2 text-sm focus:border-gray-400 focus:outline-none"
          placeholder="Ask a question about the passage…"
          value={question}
          onChange={(event) => setQuestion(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === 'Enter' && !disabled) {
              handleAsk();
            }
          }}
        />
        <button
          className="rounded-md bg-gray-900 px-4 py-2 text-sm text-white hover:bg-gray-700 disabled:cursor-not-allowed disabled:bg-gray-400"
          disabled={disabled}
          onClick={handleAsk}
        >
          Ask
        </button>
      </div>

      <pre className="min-h-[80px] whitespace-pre-wrap rounded-lg border border-gray-200 bg-white p-4 font-mono text-sm leading-relaxed">
        {output}
      </pre>

      {status && <p className="mt-2 text-xs text-gray-400">{status}</p>}
    </div>
  );
}

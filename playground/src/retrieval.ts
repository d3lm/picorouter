import MiniSearch from 'minisearch';

export interface RetrievalResult {
  document: string;
  index: number;
  score: number;
}

export interface SearchOptions {
  topK?: number;
  minScore?: number;
}

const TOKEN_REGEX = /[a-z0-9]+(?:'[a-z0-9]+)?/gi;

export function splitIntoPassages(input: string, maxWordsPerPassage = 110): string[] {
  const paragraphs = input
    .split(/\n{2,}/)
    .map((value) => value.trim())
    .filter(Boolean);

  if (paragraphs.length > 1) {
    return paragraphs;
  }

  const sentences = input
    .replace(/\s+/g, ' ')
    .trim()
    .split(/(?<=[.!?])\s+/)
    .map((value) => value.trim())
    .filter(Boolean);

  const passages: string[] = [];

  let current: string[] = [];
  let wordCount = 0;

  for (const sentence of sentences) {
    const wordsInSentence = sentence.split(/\s+/).filter(Boolean).length;

    if (wordCount + wordsInSentence > maxWordsPerPassage && current.length > 0) {
      passages.push(current.join(' ').trim());
      current = [];
      wordCount = 0;
    }

    current.push(sentence);

    wordCount += wordsInSentence;
  }

  if (current.length > 0) {
    passages.push(current.join(' ').trim());
  }

  return passages.length > 0 ? passages : [input.trim()].filter(Boolean);
}

export function search(query: string, documents: string[], options: SearchOptions = {}): RetrievalResult[] {
  const topK = options.topK ?? 3;
  const minScore = options.minScore ?? 0;

  const normalizedDocs = documents.map((doc) => doc.trim()).filter(Boolean);

  if (topK <= 0 || normalizedDocs.length === 0) {
    return [];
  }

  const queryTokens = tokenize(query);

  if (queryTokens.length === 0) {
    return normalizedDocs.slice(0, topK).map((document, index) => ({ document, index, score: 0 }));
  }

  const miniSearch = new MiniSearch<{ id: number; document: string }>({
    fields: ['document'],
    storeFields: ['document'],
    idField: 'id',
    processTerm: (term) => {
      const processed = tokenize(term).join('');
      return processed.length > 1 ? processed : null;
    },
  });

  miniSearch.addAll(normalizedDocs.map((document, index) => ({ id: index, document })));

  const queryText = queryTokens.join(' ');

  const ranked = miniSearch
    .search(queryText, {
      prefix: true,
      fuzzy: 0.2,
      combineWith: 'OR',
    })
    .map((hit) => {
      const index = Number(hit.id);
      return {
        document: normalizedDocs[index],
        index,
        score: hit.score,
      };
    })
    .filter((item) => item.document !== undefined && item.score >= minScore)
    .sort((a, b) => b.score - a.score);

  if (ranked.length === 0) {
    return normalizedDocs.slice(0, topK).map((document, index) => ({ document, index, score: 0 }));
  }

  return ranked.slice(0, topK);
}

function tokenize(text: string): string[] {
  return (text.toLowerCase().match(TOKEN_REGEX) ?? []).filter((token) => token.length > 1);
}

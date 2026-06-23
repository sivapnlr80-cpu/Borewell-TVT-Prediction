/**
 * Utility functions for chunking, vector embedding, and similarity search (RAG).
 */

/**
 * Chunks raw text into fixed size pieces with overlap.
 * @param {string} text 
 * @param {number} size 
 * @param {number} overlap 
 * @returns {string[]}
 */
export const chunkText = (text, size = 600, overlap = 150) => {
  if (!text) return [];
  const chunks = [];
  let start = 0;
  
  // Clean whitespace
  const cleanText = text.replace(/\s+/g, ' ');
  
  while (start < cleanText.length) {
    const end = Math.min(start + size, cleanText.length);
    chunks.push(cleanText.slice(start, end).trim());
    if (end === cleanText.length) break;
    start += size - overlap;
  }
  
  return chunks.filter(c => c.length > 15);
};

/**
 * Fetch embeddings using Gemini API.
 */
const getGeminiEmbeddings = async (texts, apiKey, model = 'text-embedding-004') => {
  const url = `https://generativelanguage.googleapis.com/v1beta/models/${model}:batchEmbedContents?key=${apiKey}`;
  const requests = texts.map(text => ({
    model: `models/${model}`,
    content: { parts: [{ text }] }
  }));
  
  const response = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ requests })
  });
  
  if (!response.ok) {
    throw new Error(`Gemini embedding failed with status ${response.status}: ${response.statusText}`);
  }
  
  const data = await response.json();
  if (!data.embeddings) {
    throw new Error('No embeddings returned from Gemini API');
  }
  return data.embeddings.map(e => e.values);
};

/**
 * Fetch embeddings using OpenAI API.
 */
const getOpenAIEmbeddings = async (texts, apiKey, model = 'text-embedding-3-small') => {
  const url = 'https://api.openai.com/v1/embeddings';
  const response = await fetch(url, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${apiKey}`
    },
    body: JSON.stringify({
      input: texts,
      model: model
    })
  });
  
  if (!response.ok) {
    throw new Error(`OpenAI embedding failed with status ${response.status}: ${response.statusText}`);
  }
  
  const data = await response.json();
  if (!data.data) {
    throw new Error('No embeddings returned from OpenAI API');
  }
  return data.data.map(item => item.embedding);
};

/**
 * Main embedding helper. Batches texts in size of 50 to avoid payload size or rate limits.
 */
export const getEmbeddings = async (texts, provider, apiKey) => {
  if (!apiKey || !texts || texts.length === 0) return null;
  
  try {
    // Process in batches of 50
    const batchSize = 50;
    let allEmbeddings = [];
    
    for (let i = 0; i < texts.length; i += batchSize) {
      const batch = texts.slice(i, i + batchSize);
      let batchResult;
      
      if (provider === 'gemini') {
        batchResult = await getGeminiEmbeddings(batch, apiKey);
      } else if (provider === 'openai') {
        batchResult = await getOpenAIEmbeddings(batch, apiKey);
      } else {
        return null; // Local search fallback for Anthropic or others
      }
      
      allEmbeddings = allEmbeddings.concat(batchResult);
    }
    
    return allEmbeddings;
  } catch (error) {
    console.warn("Vector embedding API call failed, falling back to keyword search:", error);
    return null;
  }
};

/**
 * Cosine similarity between two vectors.
 */
export const cosineSimilarity = (vecA, vecB) => {
  if (!vecA || !vecB || vecA.length !== vecB.length) return 0;
  let dotProduct = 0.0;
  let normA = 0.0;
  let normB = 0.0;
  for (let i = 0; i < vecA.length; i++) {
    dotProduct += vecA[i] * vecB[i];
    normA += vecA[i] * vecA[i];
    normB += vecB[i] * vecB[i];
  }
  if (normA === 0 || normB === 0) return 0;
  return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
};

/**
 * Local TF-IDF search fallback.
 */
export const localKeywordSearch = (query, chunks, topK = 5) => {
  if (!query || !chunks || chunks.length === 0) return [];
  const queryTokens = query.toLowerCase().split(/\W+/).filter(t => t.length > 2);
  if (queryTokens.length === 0) return chunks.slice(0, topK);
  
  const scored = chunks.map(chunk => {
    const text = chunk.text.toLowerCase();
    let score = 0;
    queryTokens.forEach(token => {
      const regex = new RegExp(token.replace(/[-\/\\^$*+?.()|[\]{}]/g, '\\$&'), 'g');
      const matches = text.match(regex);
      if (matches) {
        // TF score
        score += matches.length * (1 + Math.log(matches.length));
      }
    });
    return { chunk, score };
  });
  
  return scored
    .filter(item => item.score > 0)
    .sort((a, b) => b.score - a.score)
    .slice(0, topK)
    .map(item => item.chunk);
};

/**
 * Retrieve the top K relevant chunks using vector cosine similarity or TF-IDF fallback.
 */
export const retrieveRelevantContext = async (query, chunks, provider, apiKey, topK = 5) => {
  if (!query || !chunks || chunks.length === 0) return [];
  
  // Try vector similarity search
  if (apiKey && (provider === 'gemini' || provider === 'openai')) {
    try {
      const queryEmbedArray = await getEmbeddings([query], provider, apiKey);
      if (queryEmbedArray && queryEmbedArray[0]) {
        const queryVector = queryEmbedArray[0];
        
        // Filter chunks that have valid embeddings
        const vectorChunks = chunks.filter(c => c.embedding && c.embedding.length === queryVector.length);
        if (vectorChunks.length > 0) {
          const scored = vectorChunks.map(chunk => {
            const similarity = cosineSimilarity(queryVector, chunk.embedding);
            return { chunk, similarity };
          });
          
          return scored
            .sort((a, b) => b.similarity - a.similarity)
            .slice(0, topK)
            .map(item => item.chunk);
        }
      }
    } catch (e) {
      console.warn("Vector cosine search failed, falling back to local keyword search:", e);
    }
  }
  
  // Fallback to local keyword search
  return localKeywordSearch(query, chunks, topK);
};

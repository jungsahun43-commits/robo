import type { Finding, Photo, Suggestion } from './types';

export interface FindingRepository {
  snapshot(): readonly Finding[];
  subscribe(listener: () => void): () => void;
  save(finding: Finding): void;
}

export interface AiAnalyzer {
  hazard(before: Photo, signal: AbortSignal): Promise<Suggestion>;
  compare(before: Photo, after: Photo, signal: AbortSignal): Promise<Suggestion>;
}

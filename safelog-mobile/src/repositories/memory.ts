import type { FindingRepository } from '@/contracts/interfaces';
import type { Finding } from '@/contracts/types';

export class MemoryFindingRepository implements FindingRepository {
  private findings: readonly Finding[] = [];
  private listeners = new Set<() => void>();

  snapshot = () => this.findings;
  subscribe = (listener: () => void) => {
    this.listeners.add(listener);
    return () => { this.listeners.delete(listener); };
  };

  /** Removes every finding and notifies subscribers (demo reset). */
  clear() {
    this.findings = [];
    this.listeners.forEach(listener => listener());
  }

  save(finding: Finding) {
    const exists = this.findings.some(item => item.id === finding.id);
    this.findings = exists
      ? this.findings.map(item => item.id === finding.id ? finding : item)
      : [finding, ...this.findings];
    this.listeners.forEach(listener => listener());
  }
}

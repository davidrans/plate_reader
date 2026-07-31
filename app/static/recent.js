// The recent-plates list: one row per plate, with a sighting count.
//
// Deliberately DOM-free and storage-free so it can be unit tested — this is
// where the original duplicate bug lived (the old code compared each frame's
// text against only the newest row, so one misread let every later correct
// read back in as "new").
//
// Rows are owned by a track for that track's lifetime, so a plate held in
// frame for 20 frames updates one row rather than adding 20, and the sighting
// count increments once per track rather than once per frame.

import { confusionDistance } from "./tracker.js";

export const RECENT_LIMIT = 50;

// Deliberately tight: only a single OCR-confusable substitution (cost 0.5)
// merges two rows, so "ABC123"/"ABCI23" collapse but "ABC123"/"ABD123" stay
// separate. Voting already resolves near-misses within one viewing; this only
// catches a later, separate track settling on a slightly different reading.
export const MERGE_DISTANCE = 0.5;

/**
 * Bring stored rows up to the current shape. Entries written before sighting
 * counts existed only had `ts`, and must not be discarded.
 */
export function normalizeStored(stored) {
  if (!Array.isArray(stored)) return [];
  return stored.map((e) => ({
    ...e,
    count: e.count ?? 1,
    firstSeen: e.firstSeen ?? e.ts,
    lastSeen: e.lastSeen ?? e.ts,
  }));
}

export class RecentPlates {
  constructor(entries = [], limit = RECENT_LIMIT) {
    this.entries = entries;
    this.limit = limit;
  }

  /** Exact match first, then one confusable substitution. */
  find(text) {
    const exact = this.entries.find((e) => e.text === text);
    if (exact) return exact;
    let best = null;
    let bestDistance = Infinity;
    for (const entry of this.entries) {
      const d = confusionDistance(entry.text, text, MERGE_DISTANCE);
      if (d <= MERGE_DISTANCE && d < bestDistance) {
        best = entry;
        bestDistance = d;
      }
    }
    return best;
  }

  /**
   * Record a confirmed track's current reading.
   *
   * @param track the tracker's Track — carries `entryText`, the row it owns
   * @returns the row that was written
   */
  upsert({ track, text, confidence, region }, now = Date.now()) {
    const regionName = region?.name && region.name !== "Unknown" ? region.name : "";
    let entry = track.entryText ? this.find(track.entryText) : null;

    if (entry && entry.text !== text) {
      if (confusionDistance(entry.text, text, MERGE_DISTANCE) <= MERGE_DISTANCE) {
        // Same plate, trivially different reading — keep the established text
        // instead of rewriting the row every time the vote wobbles.
        text = entry.text;
      } else {
        // The vote genuinely flipped to a different plate: correct the row.
        const other = this.find(text);
        if (other && other !== entry) {
          // That reading already has a row — fold this one into it.
          other.count += entry.count;
          other.confidence = Math.max(other.confidence, confidence, entry.confidence);
          other.firstSeen = Math.min(other.firstSeen, entry.firstSeen);
          this.entries = this.entries.filter((e) => e !== entry);
          entry = other;
        } else {
          entry.text = text;
        }
      }
    }

    if (!entry) {
      entry = this.find(text);
      if (entry) {
        text = entry.text; // adopt the established spelling
      } else {
        entry = { text, confidence, region: regionName, count: 0, firstSeen: now, lastSeen: now };
        this.entries.unshift(entry);
      }
      // First row this track has written — one sighting, however many frames follow.
      entry.count += 1;
    }

    track.entryText = entry.text;
    entry.lastSeen = now;
    entry.confidence = Math.max(entry.confidence, confidence);
    if (regionName) entry.region = regionName;

    // Most recently seen plate floats to the top.
    this.entries = [entry, ...this.entries.filter((e) => e !== entry)].slice(0, this.limit);
    return entry;
  }

  clear() {
    this.entries = [];
  }
}

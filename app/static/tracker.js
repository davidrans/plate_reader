// Temporal voting over per-frame plate reads.
//
// A single frame's OCR is unreliable: pointed at one plate for a couple of
// seconds the pipeline produces 10-20 reads, most correct and a few misread by
// a character or two. Committing each read separately fills the list with
// near-duplicates, so instead we group reads that belong to the same physical
// plate into a "track", accumulate confidence-weighted votes, and let the
// majority decide the text.
//
// Two reads belong to the same track if EITHER their boxes overlap (same plate
// in roughly the same place as last frame) OR their texts are within a couple
// of edits (same plate after a fast camera move that broke the overlap). Each
// signal covers the other's failure mode: overlap survives a misread that
// changes the text, text matching survives motion that breaks the overlap.
//
// Several tracks are alive at once, one per visible plate — readPlates()
// returns every detection in the frame, so multi-plate scenes work naturally.

export const TRACKER_DEFAULTS = {
  minVotes: 3, // votes before a track is worth showing
  trackTimeoutMs: 1000, // unmatched for this long -> the plate is gone
  iouThreshold: 0.3, // box overlap that counts as "same plate"
  maxEditDistance: 2, // fuzzy text match ceiling
};

// Characters OCR genuinely confuses with each other. A substitution inside one
// of these groups is half the cost of an unrelated one, so "ABCI23" merges into
// "ABC123" more readily than a real difference like "ABD123" would.
const CONFUSION_GROUPS = ["0OQD", "1IL", "5S", "8B", "2Z", "6G"];

const CONFUSION_OF = new Map();
for (const group of CONFUSION_GROUPS) {
  for (const ch of group) CONFUSION_OF.set(ch, group);
}

function substitutionCost(a, b) {
  if (a === b) return 0;
  return CONFUSION_OF.get(a) === CONFUSION_OF.get(b) && CONFUSION_OF.has(a) ? 0.5 : 1;
}

/**
 * Levenshtein distance where confusable substitutions cost less than a full
 * edit. Runs on every detection against every live track each frame, so it
 * bails out as soon as the whole row exceeds `ceiling`.
 */
export function confusionDistance(a, b, ceiling = Infinity) {
  if (a === b) return 0;
  if (Math.abs(a.length - b.length) > ceiling) return Infinity;

  let prev = Array.from({ length: b.length + 1 }, (_, j) => j);
  for (let i = 1; i <= a.length; i++) {
    const row = [i];
    let rowMin = i;
    for (let j = 1; j <= b.length; j++) {
      row[j] = Math.min(
        prev[j] + 1, // deletion
        row[j - 1] + 1, // insertion
        prev[j - 1] + substitutionCost(a[i - 1], b[j - 1]),
      );
      if (row[j] < rowMin) rowMin = row[j];
    }
    if (rowMin > ceiling) return Infinity;
    prev = row;
  }
  return prev[b.length];
}

/** Intersection-over-union of two {x1,y1,x2,y2} boxes. */
export function iou(a, b) {
  const x1 = Math.max(a.x1, b.x1);
  const y1 = Math.max(a.y1, b.y1);
  const x2 = Math.min(a.x2, b.x2);
  const y2 = Math.min(a.y2, b.y2);
  const overlap = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  if (overlap === 0) return 0;
  const areaA = (a.x2 - a.x1) * (a.y2 - a.y1);
  const areaB = (b.x2 - b.x1) * (b.y2 - b.y1);
  const union = areaA + areaB - overlap;
  return union > 0 ? overlap / union : 0;
}

let nextTrackId = 1;

class Track {
  constructor(result, now) {
    this.id = nextTrackId++;
    this.box = result.box;
    this.firstSeen = now;
    this.lastSeen = now;
    this.votes = new Map();
    // Text this track last wrote into the recent list, so the caller can
    // rename that row in place when the leader changes rather than adding a
    // second one. Null until the track has enough votes to be committed.
    this.entryText = null;
    // Best-looking crop seen so far, kept so the row can show a picture of the
    // plate. Held as a canvas and only encoded once the caller takes it —
    // encoding every frame would be wasteful.
    this.bestCrop = null;
    this.bestCropConfidence = -1;
    this.cropChanged = false;
    this.addVote(result, now);
  }

  /**
   * Hand over the best crop if it has improved since the last call, so the
   * caller can encode it once instead of on every frame.
   */
  takeCropIfChanged() {
    if (!this.cropChanged) return null;
    this.cropChanged = false;
    return this.bestCrop;
  }

  addVote(result, now) {
    const { text, confidence, region } = result;
    const vote = this.votes.get(text) ?? { weight: 0, count: 0, maxConf: 0, region: null };
    // Weighted by confidence: a shaky read shouldn't outvote confident ones.
    vote.weight += confidence;
    vote.count += 1;
    if (confidence >= vote.maxConf) {
      vote.maxConf = confidence;
      vote.region = region ?? vote.region;
    }
    this.votes.set(text, vote);

    // Keep the clearest view of the plate, not merely the most recent one.
    if (result.crop && confidence > this.bestCropConfidence) {
      this.bestCrop = result.crop;
      this.bestCropConfidence = confidence;
      this.cropChanged = true;
    }

    this.box = result.box;
    this.lastSeen = now;
  }

  /** The winning reading so far: highest total confidence-weighted vote. */
  leader() {
    let bestText = null;
    let best = null;
    for (const [text, vote] of this.votes) {
      if (!best || vote.weight > best.weight) {
        best = vote;
        bestText = text;
      }
    }
    return {
      text: bestText,
      weight: best.weight,
      confidence: best.maxConf,
      region: best.region,
    };
  }

  totalVotes() {
    let total = 0;
    for (const vote of this.votes.values()) total += vote.count;
    return total;
  }
}

export class PlateTracker {
  constructor(options = {}) {
    this.opts = { ...TRACKER_DEFAULTS, ...options };
    this.tracks = [];
  }

  /**
   * Fold one frame's reads into the live tracks.
   *
   * @param results output of readPlates()
   * @param now timestamp in ms (injectable so tests are deterministic)
   * @returns {{live: Array, confirmed: Array, expired: Array}} — `confirmed`
   *   are tracks with enough votes to belong in the list, each carrying its
   *   current leader; the caller upserts them.
   */
  update(results, now = Date.now()) {
    // One detection per track per frame. Without this, two different cars whose
    // plates happen to read alike would collapse into a single track — the
    // second detection would text-match the track the first just claimed, even
    // though their boxes are nowhere near each other.
    const claimed = new Set();
    for (const result of results) {
      const match = this.findTrack(result, claimed);
      if (match) {
        match.addVote(result, now);
        claimed.add(match);
      } else {
        const track = new Track(result, now);
        this.tracks.push(track);
        claimed.add(track);
      }
    }

    const expired = this.tracks.filter((t) => now - t.lastSeen > this.opts.trackTimeoutMs);
    this.tracks = this.tracks.filter((t) => now - t.lastSeen <= this.opts.trackTimeoutMs);

    const live = this.tracks.map((t) => ({ track: t, ...t.leader(), votes: t.totalVotes() }));
    const confirmed = live.filter((t) => t.votes >= this.opts.minVotes);
    return { live, confirmed, expired };
  }

  // A detection joins a track on box overlap OR text similarity. Where several
  // tracks qualify, prefer the strongest box overlap — with two similar plates
  // on screen at once, position is the more trustworthy signal.
  findTrack(result, claimed = new Set()) {
    let best = null;
    let bestIou = 0;
    let textMatch = null;

    for (const track of this.tracks) {
      if (claimed.has(track)) continue;
      const overlap = iou(result.box, track.box);
      if (overlap >= this.opts.iouThreshold && overlap > bestIou) {
        best = track;
        bestIou = overlap;
      }
      if (
        !textMatch &&
        confusionDistance(result.text, track.leader().text, this.opts.maxEditDistance) <=
          this.opts.maxEditDistance
      ) {
        textMatch = track;
      }
    }
    return best ?? textMatch;
  }

  reset() {
    this.tracks = [];
  }
}

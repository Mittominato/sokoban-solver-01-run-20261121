// Room-by-room push-optimal Sokoban solver.
//
// The levels this competition draws are a hallway with small rooms off it
// (3-4 boxes each, one door onto the hallway). A box pushed out of a room
// into the hallway can never come back, so every box stays in its room and
// the level's push optimum is the sum of its rooms' optima. This program
// therefore splits each level into rooms, solves every room on its own with
// a push-based A* search, and stitches the room solutions together with
// walks along the hallway.
//
// The search works on pushes, not steps: a state is the boxes' squares plus
// the player's reachable region (named by its lowest square), so walking is
// free and only pushes cost. The estimate is the cheapest assignment of
// boxes to goals under walking distances from the goals, which never
// overestimates the pushes left, so the first solved state A* pops is
// push-optimal for its room. Dead squares (found by pulling boxes back from
// the goals), frozen 2x2 blocks and boxes walled in on all four sides prune
// states no push can save. Every budget is a count of expanded states, never
// the clock, so two runs write the same bytes.
//
// Node 22 runs this file as it is (erasable TypeScript syntax only). It reads
// /input/<id>/level.txt and writes /output/<id>/solution.txt for each id, in
// one run over the whole set. No network, no clock reads, no randomness.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

// The four steps: up, down, left, right, with the move letter each writes.
const DX = [0, 0, -1, 1];
const DY = [-1, 1, 0, 0];
const LETTER = ['u', 'd', 'l', 'r'];

// How far one room search may go before it gives up, and how far the retry
// may go. Counts of expanded states: deterministic, never the clock.
const ROOM_BUDGET = 250_000;
const ROOM_RETRY_BUDGET = 1_000_000;
const WHOLE_LEVEL_BUDGET = 600_000;

/** A parsed level: width, height, walls, goals, boxes and the player. */
interface Level {
  w: number;
  h: number;
  wall: Uint8Array;
  goals: number[];
  boxes: number[];
  player: number;
}

/** Parse XSB text (# wall, space floor, . goal, $ box, * box on goal, @ player). */
function parseLevel(text: string): Level {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const h = rows.length;
  const w = Math.max(...rows.map((r) => r.length));
  const wall = new Uint8Array(w * h).fill(1);
  const goals: number[] = [];
  const boxes: number[] = [];
  let player = -1;
  rows.forEach((row, y) => {
    for (let x = 0; x < row.length; x++) {
      const ch = row[x];
      const i = y * w + x;
      if (ch === '#') continue;
      wall[i] = 0;
      if (ch === '.' || ch === '*' || ch === '+') goals.push(i);
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  });
  // The outermost frame is wall whatever it shows: nothing leaves the board.
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (x === 0 || y === 0 || x === w - 1 || y === h - 1) wall[y * w + x] = 1;
    }
  }
  return { w, h, wall, goals, boxes: boxes.sort((a, b) => a - b), player };
}

/**
 * The live squares: every square from which a box, pushed on its own, can
 * still reach a goal. Found by pulling boxes back from the goals: a box on
 * square y can be pushed onto square x = y + d when the player can stand on
 * y - d, so from a live square x every neighbour y with y - d open floor is
 * live too. A box off a live square is dead, and any state holding one is
 * pruned.
 */
function liveSquares(level: Level): Uint8Array {
  const { w, wall, goals } = level;
  const live = new Uint8Array(wall.length);
  const queue = [...goals];
  for (const g of goals) live[g] = 1;
  for (let qi = 0; qi < queue.length; qi++) {
    const x = queue[qi];
    for (let d = 0; d < 4; d++) {
      const y = x - (DY[d] * w + DX[d]);
      const stand = y - (DY[d] * w + DX[d]);
      if (y < 0 || y >= wall.length || stand < 0 || stand >= wall.length) continue;
      if (!wall[y] && !wall[stand] && !live[y]) {
        live[y] = 1;
        queue.push(y);
      }
    }
  }
  return live;
}

/**
 * The squares the player reaches from `from` with `occupied` in the way,
 * flood-filled over `floor`; returns the region's lowest square, which names
 * the region. `seen`/`stamp`/`stack` are scratch space the caller owns.
 */
function regionLow(
  w: number,
  floor: Uint8Array,
  occupied: Uint8Array,
  from: number,
  seen: Int32Array,
  stamp: number,
  stack: Int32Array,
): number {
  let low = from;
  let n = 0;
  stack[n++] = from;
  seen[from] = stamp;
  while (n > 0) {
    const p = stack[--n];
    if (p < low) low = p;
    for (let d = 0; d < 4; d++) {
      const q = p + DY[d] * w + DX[d];
      if (floor[q] && !occupied[q] && seen[q] !== stamp) {
        seen[q] = stamp;
        stack[n++] = q;
      }
    }
  }
  return low;
}

/** Whether `cell` holds a box in `occupied`. Kept as a helper for clarity. */
function isBox(occupied: Uint8Array, cell: number): boolean {
  return occupied[cell] === 1;
}

/**
 * The shortest walk from `from` to `to` over `floor` with `occupied` in the
 * way, as lower-case LURD letters (up first on ties, then down, left,
 * right), or null when `to` is out of reach. The walk never pushes a box.
 */
function walkTo(
  w: number,
  floor: Uint8Array,
  occupied: Uint8Array,
  from: number,
  to: number,
): string | null {
  if (from === to) return '';
  const prev = new Int32Array(floor.length).fill(-1);
  const how = new Int8Array(floor.length).fill(-1);
  const queue = [from];
  prev[from] = from;
  for (let qi = 0; qi < queue.length; qi++) {
    const p = queue[qi];
    if (p === to) break;
    for (let d = 0; d < 4; d++) {
      const q = p + DY[d] * w + DX[d];
      if (!floor[q] || occupied[q] || prev[q] !== -1) continue;
      prev[q] = p;
      how[q] = d;
      queue.push(q);
    }
  }
  if (prev[to] === -1) return null;
  let out = '';
  for (let p = to; p !== from; p = prev[p]) out = LETTER[how[p]] + out;
  return out;
}

/** All orderings of 0..n-1, for the assignment heuristic (n is at most 4). */
function permutations(n: number): number[][] {
  if (n === 0) return [[]];
  const rest = permutations(n - 1);
  const out: number[][] = [];
  for (const r of rest) {
    for (let i = 0; i <= r.length; i++) {
      const copy = r.slice();
      copy.splice(i, 0, n - 1);
      out.push(copy);
    }
  }
  return out;
}

/**
 * A push-based A* search over one board.
 *
 * `floor` marks the squares the player and boxes may stand on, `goals` the
 * goal squares, `startBoxes` the boxes (sorted) and `startPlayer` the
 * player's square. `boxFloor` marks the squares a box may be pushed onto
 * (a subset of `floor`: for a room search the room's own squares, never the
 * hallway). Returns the push-optimal move string with the player ending
 * wherever the last push left them, plus that end square, or null when the
 * search gives up within `maxExpansions` expanded states.
 */
function pushAStar(
  w: number,
  floor: Uint8Array,
  goals: number[],
  startBoxes: number[],
  startPlayer: number,
  boxFloor: Uint8Array,
  maxExpansions: number,
): { moves: string; endPlayer: number; pushes: number } | null {
  const size = floor.length;
  const n = goals.length;
  if (startBoxes.length !== n) return null;
  const goalSet = new Uint8Array(size);
  for (const g of goals) goalSet[g] = 1;

  // Walking distances from each goal over the floor alone, for the heuristic.
  const dist: Int32Array[] = goals.map((g) => {
    const d = new Int32Array(size).fill(-1);
    const queue = [g];
    d[g] = 0;
    for (let qi = 0; qi < queue.length; qi++) {
      const p = queue[qi];
      for (let dir = 0; dir < 4; dir++) {
        const q = p + DY[dir] * w + DX[dir];
        if (floor[q] && d[q] < 0) {
          d[q] = d[p] + 1;
          queue.push(q);
        }
      }
    }
    return d;
  });

  // Dead squares on this board, for pruning.
  const live = liveSquares({ w, h: size / w, wall: invert(floor), goals, boxes: [], player: -1 });
  if (startBoxes.some((b) => !live[b])) return null;

  const orderings = permutations(n);
  // The cheapest assignment of the boxes to the goals, or Infinity when no
  // assignment reaches every goal: an admissible estimate of pushes left.
  const estimate = (boxes: number[]): number => {
    let best = Infinity;
    for (const perm of orderings) {
      let sum = 0;
      let ok = true;
      for (let i = 0; i < n; i++) {
        const d = dist[perm[i]][boxes[i]];
        if (d < 0) {
          ok = false;
          break;
        }
        sum += d;
      }
      if (ok && sum < best) best = sum;
    }
    return best;
  };

  const solved = (boxes: number[]): boolean => boxes.every((b) => goalSet[b]);

  // Scratch space shared by the region floods: `seen` marks the region of
  // the state being expanded, `mark` the regions of the states its pushes
  // make (kept apart, so marking a successor never erases the parent's).
  const occupied = new Uint8Array(size);
  const seen = new Int32Array(size);
  const mark = new Int32Array(size);
  const stack = new Int32Array(size);
  let stamp = 0;
  let markStamp = 0;

  // Whether the box at `t` sits in a frozen 2x2 block of walls and boxes
  // with a box off its goal: no push can ever free it.
  const frozenBlock = (t: number): boolean => {
    for (const a of [t, t - 1, t - w, t - w - 1]) {
      const b = a + 1;
      const c = a + w;
      const d = c + 1;
      if (a < 0 || d >= size) continue;
      const filled =
        (isWall(floor, a) || isBox(occupied, a)) &&
        (isWall(floor, b) || isBox(occupied, b)) &&
        (isWall(floor, c) || isBox(occupied, c)) &&
        (isWall(floor, d) || isBox(occupied, d));
      if (!filled) continue;
      if (
        (isBox(occupied, a) && !goalSet[a]) ||
        (isBox(occupied, b) && !goalSet[b]) ||
        (isBox(occupied, c) && !goalSet[c]) ||
        (isBox(occupied, d) && !goalSet[d])
      ) {
        return true;
      }
    }
    return false;
  };

  // Whether the box at `b` can never move again: every push needs a floor
  // square to land on and a floor square for the player to stand on, and
  // walls alone block all four. (Other boxes may move away, walls cannot.)
  const walledIn = (b: number): boolean => {
    if (goalSet[b]) return false;
    for (let dir = 0; dir < 4; dir++) {
      const step = DY[dir] * w + DX[dir];
      if (floor[b + step] && floor[b - step]) return false;
    }
    return true;
  };

  // The states: boxes, region low, parent, and the push that made the state.
  const stateBoxes: number[][] = [startBoxes.slice()];
  const stateLow: number[] = [0];
  const parent: number[] = [-1];
  const pushFrom: number[] = [-1];
  const pushDir: number[] = [-1];
  const gScore: number[] = [0];
  const fScore: number[] = [0];
  const keyOf = (boxes: number[], low: number): string => `${boxes.join(',')}|${low}`;

  const best = new Map<string, number>();
  for (const b of startBoxes) occupied[b] = 1;
  stateLow[0] = regionLow(w, floor, occupied, startPlayer, seen, ++stamp, stack);
  for (const b of startBoxes) occupied[b] = 0;
  const startEstimate = estimate(startBoxes);
  if (startEstimate === Infinity) return null;
  fScore[0] = startEstimate;
  best.set(keyOf(startBoxes, stateLow[0]), 0);

  // A binary heap of state indices, ordered by f, then by deeper g, then by
  // the earlier state: fully deterministic ties.
  const heap: number[] = [0];
  const heapLess = (a: number, b: number): boolean =>
    fScore[a] < fScore[b] ||
    (fScore[a] === fScore[b] && (gScore[a] > gScore[b] || (gScore[a] === gScore[b] && a < b)));
  const heapPush = (s: number): void => {
    let i = heap.length;
    heap.push(s);
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!heapLess(heap[i], heap[p])) break;
      [heap[i], heap[p]] = [heap[p], heap[i]];
      i = p;
    }
  };
  const heapPop = (): number => {
    const top = heap[0];
    const last = heap.pop()!;
    if (heap.length > 0) {
      heap[0] = last;
      for (let i = 0; ; ) {
        let c = 2 * i + 1;
        if (c >= heap.length) break;
        if (c + 1 < heap.length && heapLess(heap[c + 1], heap[c])) c++;
        if (!heapLess(heap[c], heap[i])) break;
        [heap[i], heap[c]] = [heap[c], heap[i]];
        i = c;
      }
    }
    return top;
  };

  if (solved(startBoxes)) return { moves: '', endPlayer: startPlayer, pushes: 0 };

  let expanded = 0;
  let found = -1;
  while (heap.length > 0 && expanded < maxExpansions) {
    const s = heapPop();
    if (best.get(keyOf(stateBoxes[s], stateLow[s])) !== s) continue;
    const boxes = stateBoxes[s];
    if (solved(boxes)) {
      found = s;
      break;
    }
    expanded++;
    for (const b of boxes) occupied[b] = 1;
    const here = ++stamp;
    regionLow(w, floor, occupied, stateLow[s], seen, here, stack);
    for (let bi = 0; bi < boxes.length; bi++) {
      const b = boxes[bi];
      for (let dir = 0; dir < 4; dir++) {
        const step = DY[dir] * w + DX[dir];
        const stand = b - step;
        const target = b + step;
        if (seen[stand] !== here) continue;
        if (!boxFloor[target] || occupied[target] || !live[target]) continue;
        occupied[b] = 0;
        occupied[target] = 1;
        const dead = walledIn(target) || frozenBlock(target);
        let next: number[] | null = null;
        let nextLow = -1;
        let h = Infinity;
        if (!dead) {
          next = boxes.slice();
          next[bi] = target;
          next.sort((a, b2) => a - b2);
          h = estimate(next);
          if (h !== Infinity) {
            nextLow = regionLow(w, floor, occupied, b, mark, ++markStamp, stack);
          }
        }
        occupied[target] = 0;
        occupied[b] = 1;
        if (dead || h === Infinity) continue;
        const key = keyOf(next!, nextLow);
        const g = gScore[s] + 1;
        const known = best.get(key);
        if (known !== undefined && gScore[known] <= g) continue;
        const t = stateBoxes.length;
        stateBoxes.push(next!);
        stateLow.push(nextLow);
        parent.push(s);
        pushFrom.push(b);
        pushDir.push(dir);
        gScore.push(g);
        fScore.push(g + h);
        best.set(key, t);
        heapPush(t);
      }
    }
    for (const b of boxes) occupied[b] = 0;
  }
  if (found < 0) return null;

  // The pushes back to the start, then the moves: walk to each push's
  // square, then push.
  const path: number[] = [];
  for (let s = found; s > 0; s = parent[s]) path.push(s);
  path.reverse();
  let moves = '';
  let player = startPlayer;
  let boxes = startBoxes.slice();
  for (const s of path) {
    const from = pushFrom[s];
    const dir = pushDir[s];
    const step = DY[dir] * w + DX[dir];
    for (const b of boxes) occupied[b] = 1;
    const walk = walkTo(w, floor, occupied, player, from - step);
    for (const b of boxes) occupied[b] = 0;
    if (walk === null) return null;
    moves += walk + LETTER[dir];
    player = from;
    boxes = boxes.map((b) => (b === from ? from + step : b)).sort((a, b2) => a - b2);
  }
  return { moves, endPlayer: player, pushes: path.length };
}

/** `floor` with its bits flipped: the walls of the same board. */
function invert(floor: Uint8Array): Uint8Array {
  const out = new Uint8Array(floor.length);
  for (let i = 0; i < floor.length; i++) out[i] = floor[i] ? 0 : 1;
  return out;
}

/** Whether `cell` is a wall on a board whose open squares are `floor`. */
function isWall(floor: Uint8Array, cell: number): boolean {
  return cell < 0 || cell >= floor.length || floor[cell] === 0;
}

/** One room: its squares, boxes, goals, and the hallway square outside its door. */
interface Room {
  squares: number[];
  boxes: number[];
  goals: number[];
  entry: number;
}

/**
 * The level's rooms. The hallway is the player's row: a box there reaches
 * no goal, so its dead squares cut the board and each connected part left
 * that holds boxes is a room. Null when the level is no set of rooms (a box
 * starts in the hallway, a goal falls outside every part, or a part's boxes
 * and goals do not match).
 */
function roomsOf(level: Level): Room[] | null {
  const { w, wall, goals, boxes, player } = level;
  const size = wall.length;
  const live = liveSquares(level);
  const row = Math.floor(player / w);
  const cut = new Uint8Array(size);
  for (let x = 0; x < w; x++) {
    const i = row * w + x;
    if (!wall[i] && !live[i]) cut[i] = 1;
  }
  if (boxes.some((b) => cut[b])) return null;
  const part = new Int32Array(size).fill(-1);
  const rooms: Room[] = [];
  for (let s = 0; s < size; s++) {
    if (wall[s] || cut[s] || part[s] >= 0) continue;
    const squares = [s];
    part[s] = rooms.length;
    for (let i = 0; i < squares.length; i++) {
      for (let d = 0; d < 4; d++) {
        const q = squares[i] + DY[d] * w + DX[d];
        if (!wall[q] && !cut[q] && part[q] < 0) {
          part[q] = rooms.length;
          squares.push(q);
        }
      }
    }
    rooms.push({ squares, boxes: [], goals: [], entry: -1 });
  }
  for (const b of boxes) rooms[part[b]].boxes.push(b);
  for (const g of goals) {
    if (part[g] < 0) return null;
    rooms[part[g]].goals.push(g);
  }
  if (rooms.some((r) => r.boxes.length !== r.goals.length)) return null;
  const withBoxes = rooms.filter((r) => r.boxes.length > 0);
  // Each room's entry: the hallway square beside its door square, the room
  // square touching a floor square of the hallway row.
  for (const room of withBoxes) {
    const inRoom = new Uint8Array(size);
    for (const s of room.squares) inRoom[s] = 1;
    for (const s of room.squares) {
      for (let d = 0; d < 4; d++) {
        const q = s + DY[d] * w + DX[d];
        if (Math.floor(q / w) === row && !wall[q] && !inRoom[q]) {
          room.entry = q;
          break;
        }
      }
      if (room.entry >= 0) break;
    }
    if (room.entry < 0) return null;
  }
  return withBoxes;
}

/** Solve one level, room by room, and return its moves ('' when it gives up). */
function solveLevel(text: string): string {
  const level = parseLevel(text);
  const { w, wall } = level;
  const size = wall.length;
  const rooms = roomsOf(level);
  if (rooms === null) return solveWhole(level);

  // The floor every room search walks on: the whole board with walls alone.
  const open = new Uint8Array(size);
  for (let i = 0; i < size; i++) open[i] = wall[i] ? 0 : 1;

  let player = level.player;
  let boxes = level.boxes.slice();
  let moves = '';
  for (const room of rooms) {
    // Walk the player to the room's door on the whole board, boxes as walls.
    const occupied = new Uint8Array(size);
    for (const b of boxes) occupied[b] = 1;
    const walk = walkTo(w, open, occupied, player, room.entry);
    if (walk === null) return '';
    moves += walk;
    player = room.entry;

    // The room's own board: its squares plus the entry square; a box may
    // only ever stand on the room's squares, never the hallway.
    const floor = new Uint8Array(size);
    for (const s of room.squares) floor[s] = 1;
    floor[room.entry] = 1;
    const boxFloor = new Uint8Array(size);
    for (const s of room.squares) boxFloor[s] = 1;
    const roomBoxes = room.boxes.slice().sort((a, b) => a - b);
    const roomGoals = room.goals.slice().sort((a, b) => a - b);
    let solved = pushAStar(w, floor, roomGoals, roomBoxes, player, boxFloor, ROOM_BUDGET);
    if (solved === null) {
      solved = pushAStar(w, floor, roomGoals, roomBoxes, player, boxFloor, ROOM_RETRY_BUDGET);
    }
    if (solved === null) return '';
    moves += solved.moves;
    player = solved.endPlayer;
    // The room's boxes now stand on its goals.
    boxes = boxes.filter((b) => !room.boxes.includes(b));
    for (const g of roomGoals) boxes.push(g);
    boxes.sort((a, b) => a - b);
  }
  return moves;
}

/** Fallback: solve the whole level as one board, when it is no set of rooms. */
function solveWhole(level: Level): string {
  const { w, wall } = level;
  const size = wall.length;
  const floor = new Uint8Array(size);
  for (let i = 0; i < size; i++) floor[i] = wall[i] ? 0 : 1;
  const solved = pushAStar(
    w,
    floor,
    level.goals.slice().sort((a, b) => a - b),
    level.boxes.slice().sort((a, b) => a - b),
    level.player,
    floor,
    WHOLE_LEVEL_BUDGET,
  );
  return solved === null ? '' : solved.moves;
}

for (const id of readdirSync(INPUT_DIR).sort()) {
  let moves = '';
  try {
    moves = solveLevel(readFileSync(join(INPUT_DIR, id, 'level.txt'), 'utf8'));
  } catch {
    moves = '';
  }
  mkdirSync(join(OUTPUT_DIR, id), { recursive: true });
  writeFileSync(join(OUTPUT_DIR, id, 'solution.txt'), `${moves}\n`);
}

/*
 * Landing page backdrop: an adaptive octree searching for a hypocentre.
 *
 * Each cycle replays one detection. Nature comes first, soft and warm: an
 * earthquake ruptures a patch of the fault and its P and S waves ripple out
 * to the surface and the stations. Then Qseek, crisp and geometric: the root
 * nodes light up with the stacked semblance, the octree refines the most
 * coherent nodes level by level (pruning a ghost maximum on the way), locks
 * onto the hypocentre and migrates the rays back to the stations. Located
 * events accumulate into a catalog along a fault.
 */
(() => {
  "use strict";

  const NS = "http://www.w3.org/2000/svg";
  const HALF = [ 1, 1, 2 / 3 ];  // half extents of the search volume
  const ROOT = [ 6, 6, 4 ];      // root nodes per axis
  const SPLITS = [ 6, 5, 4, 3 ]; // nodes refined per level
  const LEVELS = SPLITS.length;
  const N_STATIONS = 13;
  const DOT_BUCKETS = 5;
  const DEPTH_BUCKETS = 3;
  const CATALOG_SIZE = 46;
  const V_P = 0.85;
  const V_S = V_P / 1.75;
  const CAMERA = 5.2;
  const RIPPLES = [ 1, 0.42, 0.16 ]; // opacity of a crest and its ripples
  const WAVELENGTH = 0.075;
  const EDGES = [
    [ 0, 1 ],
    [ 2, 3 ],
    [ 4, 5 ],
    [ 6, 7 ],
    [ 0, 2 ],
    [ 1, 3 ],
    [ 4, 6 ],
    [ 5, 7 ],
    [ 0, 4 ],
    [ 1, 5 ],
    [ 2, 6 ],
    [ 3, 7 ],
  ];

  // Timeline of a single event, in seconds.
  const T = {
    stack : 1,
    split : 2.6,
    step : 0.95,
    grow : 0.6,
  };
  T.lock = T.split + LEVELS * T.step + 0.1;
  T.rays = T.lock + 0.35;
  T.catalog = T.lock + 0.9;
  T.fade = T.lock + 3.1;
  T.end = T.fade + 1.4;

  const rand = (a, b) => a + Math.random() * (b - a);
  const gauss = () => {
    let u = 0;
    while (!u)
      u = Math.random();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * Math.random());
  };
  const clamp = (x, a, b) => Math.min(b, Math.max(a, x));
  const ease = (x) => (x <= 0 ? 0 : x >= 1 ? 1 : 1 - (1 - x) ** 3);
  const smooth = (a, b, x) => {
    const u = clamp((x - a) / (b - a), 0, 1);
    return u * u * (3 - 2 * u);
  };
  const lerp3 = (p, q, u) => [p[0] + (q[0] - p[0]) * u,
                              p[1] + (q[1] - p[1]) * u,
                              p[2] + (q[2] - p[2]) * u,
  ];
  const dist2 = (p, q) =>
      (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 + (p[2] - q[2]) ** 2;
  const r1 = (v) => Math.round(v * 10) / 10;

  const el = (tag, attrs, parent) => {
    const node = document.createElementNS(NS, tag);
    for (const key in attrs)
      node.setAttribute(key, attrs[key]);
    if (parent)
      parent.appendChild(node);
    return node;
  };

  /** A dipping fault plane with a migrating swarm on it. */
  class Fault {
    constructor() {
      const strike = rand(0, Math.PI);
      const dip = rand(55, 75) * (Math.PI / 180);
      this.s = [ Math.cos(strike), Math.sin(strike), 0 ];
      this.d = [
        -Math.sin(strike) * Math.cos(dip),
        Math.cos(strike) * Math.cos(dip),
        -Math.sin(dip),
      ];
      this.n = [
        this.s[1] * this.d[2] - this.s[2] * this.d[1],
        this.s[2] * this.d[0] - this.s[0] * this.d[2],
        this.s[0] * this.d[1] - this.s[1] * this.d[0],
      ];
      this.front = rand(-0.4, 0.4);
      this.heading = Math.random() < 0.5 ? -1 : 1;
    }

    /** Advance the swarm front along strike, bouncing at the tips. */
    migrate() {
      this.front += this.heading * rand(0.02, 0.12);
      if (Math.abs(this.front) > 0.5)
        this.heading *= -1;
    }

    sample(spread = 0.16) {
      const a = this.front + gauss() * spread;
      const b = gauss() * 0.24;
      const c = gauss() * 0.025;
      return [ 0, 1, 2 ].map(
          (i) => clamp(
              this.s[i] * a + this.d[i] * b + this.n[i] * c -
                  (i === 2 ? 0.08 : 0),
              -HALF[i] * 0.78,
              HALF[i] * 0.78,
              ),
      );
    }
  }

  class OctreeBackdrop {
    constructor(host) {
      this.host = host;
      this.reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
      this.clock = 0;
      this.yaw0 = rand(0, 2 * Math.PI);
      this.yawDrift = rand(0, 100);
      this.visible = true;
      this.last = 0;
      this.raf = 0;
      this.eventNo = 0;
      this.loop = this.loop.bind(this);

      this.fault = new Fault();
      this.stations = this.makeStations();
      this.roots = this.makeRoots();
      this.catalog = [];
      for (let i = 0; i < 26; i++) {
        this.fault.migrate();
        this.catalog.push(
            {p : this.fault.sample(0.3), no : -rand(1, CATALOG_SIZE)});
      }

      this.build();
      this.startEvent();
      this.resize();

      this.resizer = new ResizeObserver(() => this.resize());
      this.resizer.observe(host);

      if (this.reduced) {
        // A still frame of a converged search.
        for (let t = 0; t < T.rays + 1.6; t += 0.05)
          this.tick(0.05);
        this.draw();
        return;
      }

      this.observer = new IntersectionObserver(([ entry ]) => {
        this.visible = entry.isIntersecting;
        if (this.visible)
          this.request();
      });
      this.observer.observe(host);
      this.request();
    }

    makeStations() {
      const stations = [];
      for (let tries = 0; stations.length < N_STATIONS && tries < 500;
           tries++) {
        const p = [ rand(-1.15, 1.15), rand(-1.15, 1.15), HALF[2] ];
        if (stations.every((q) => dist2(p, q.p) > 0.33 ** 2)) {
          stations.push({p, hot : 0});
        }
      }
      return stations;
    }

    makeRoots() {
      const roots = [];
      const size = (2 * HALF[0]) / ROOT[0];
      for (let i = 0; i < ROOT[0]; i++)
        for (let j = 0; j < ROOT[1]; j++)
          for (let k = 0; k < ROOT[2]; k++) {
            roots.push({
              c : [
                -HALF[0] + (i + 0.5) * size,
                -HALF[1] + (j + 0.5) * size,
                -HALF[2] + (k + 0.5) * size,
              ],
              h : size / 2,
              level : 0,
            });
          }
      return roots;
    }

    build() {
      const svg = (this.svg = el("svg", {
                     class : "qs-octree",
                     "aria-hidden" : "true",
                     focusable : "false",
                   }));
      const defs = el("defs", {}, svg);
      const glow = el("radialGradient", {id : "qs-octree-glow"}, defs);
      el("stop",
         {offset : "0", class : "oct-glow-stop", "stop-opacity" : "0.55"},
         glow);
      el("stop",
         {offset : "0.35", class : "oct-glow-stop", "stop-opacity" : "0.16"},
         glow);
      el("stop", {offset : "1", class : "oct-glow-stop", "stop-opacity" : "0"},
         glow);
      const warm = el("radialGradient", {id : "qs-octree-quake"}, defs);
      for (const [offset, o] of [[ 0, 0.5 ], [ 0.4, 0.14 ], [ 1, 0 ]]) {
        el("stop", {offset, class : "oct-nature-stop", "stop-opacity" : o},
           warm);
      }

      const path = (cls, attrs = {}) =>
          el("path", {class : cls, d : "", ...attrs}, svg);

      this.surface = path("oct-surface");
      this.frame = path("oct-frame");
      this.catalogPaths = [ 0.7, 0.4, 0.18 ].map(
          (o) => path("oct-catalog", {"stroke-opacity" : o}),
      );

      this.dotPaths = [];
      for (let level = 0; level <= LEVELS; level++) {
        const row = [];
        for (let b = 0; b < DOT_BUCKETS; b++) {
          const w = [ 3.4, 2.8, 2.3, 1.9, 1.6 ][level] *
                    (0.75 + (0.5 * b) / DOT_BUCKETS);
          row.push(
              path(b === DOT_BUCKETS - 1 ? "oct-dot oct-dot--hot" : "oct-dot", {
                "stroke-width" : w.toFixed(2),
                "stroke-opacity" : [ 0.22, 0.38, 0.55, 0.75, 0.95 ][b],
              }),
          );
        }
        this.dotPaths.push(row);
      }

      this.cubes = el("g", {class : "oct-cubes"}, svg);
      this.prunedPath =
          el("path", {class : "oct-cube oct-cube--pruned", d : ""}, this.cubes);
      this.cubePaths = [];
      for (let level = 1; level <= LEVELS; level++) {
        const row = [];
        for (let b = 0; b < DEPTH_BUCKETS; b++) {
          const o = [ 0, 0.34, 0.44, 0.56, 0.72 ][level] * [ 1, 0.7, 0.42 ][b];
          row.push(
              el(
                  "path",
                  {
                    class : level === LEVELS ? "oct-cube oct-cube--fine"
                                             : "oct-cube",
                    d : "",
                    "stroke-opacity" : o.toFixed(3),
                  },
                  this.cubes,
                  ),
          );
        }
        this.cubePaths.push(row);
      }

      // Wavefronts: a soft crest followed by fading ripples.
      const wave = (cls) => {
        const g = el("g", {class : `oct-waves ${cls}`, opacity : 0}, svg);
        const glow = el("path", {class : "oct-wave-glow", d : ""}, g);
        const crests = RIPPLES.map(
            (o) => el("path",
                      {class : "oct-wave", d : "", "stroke-opacity" : o}, g));
        return {g, glow, crests};
      };
      this.waveP = wave("oct-waves--p");
      this.waveS = wave("oct-waves--s");

      this.quake = el("g", {class : "oct-quake", opacity : 0}, svg);
      this.quakeGlow =
          el("circle", {r : 0, fill : "url(#qs-octree-quake)"}, this.quake);
      this.quakePatch =
          el("path", {class : "oct-quake-patch", d : ""}, this.quake);
      this.quakeCore =
          el("circle", {class : "oct-quake-core", r : 0}, this.quake);

      this.rays = el("g", {class : "oct-rays"}, svg);
      for (const st of this.stations) {
        st.ray =
            el("path", {class : "oct-ray", d : "", pathLength : 1}, this.rays);
      }
      this.drop = path("oct-drop");
      for (const st of this.stations)
        st.el = path("oct-station");

      this.estimate = el("g", {class : "oct-estimate"}, svg);
      this.estRing = el("circle", {r : 9}, this.estimate);
      this.estTicks = el("path", {d : ""}, this.estimate);

      this.hypo = el("g", {class : "oct-hypo"}, svg);
      this.hypoGlow =
          el("circle", {r : 0, fill : "url(#qs-octree-glow)"}, this.hypo);
      this.pulses = [ 0, 1 ].map(
          () => el("circle", {class : "oct-pulse", r : 0}, this.hypo));
      this.hypoCore = el("circle", {class : "oct-hypo-core", r : 0}, this.hypo);

      this.host.appendChild(svg);
    }

    semblance(p) {
      const s = Math.exp(-dist2(p, this.h) / (2 * 0.2 ** 2));
      const g =
          this.ghostAmp * Math.exp(-dist2(p, this.ghost) / (2 * 0.3 ** 2));
      return Math.max(s, g);
    }

    startEvent() {
      this.eventNo += 1;
      this.t = 0;
      this.pause = rand(0.2, 1.4);
      this.fault.migrate();
      this.h = this.fault.sample();
      // Heterogeneities of the crust that bend the wavefronts.
      this.wobble = [ 0, 1, 2, 3 ].map(() => {
        const k = [ gauss(), gauss(), gauss() ];
        const len = Math.hypot(...k);
        return {
          k : k.map((x) => x / len),
          freq : rand(2, 5.5),
          phase : rand(0, 2 * Math.PI),
          amp : rand(0.01, 0.025),
        };
      });
      do {
        this.ghost = [ 0, 1, 2 ].map((i) => rand(-HALF[i], HALF[i]) * 0.8);
      } while (dist2(this.ghost, this.h) < 0.75 ** 2);
      this.ghostAmp = rand(0.5, 0.82);

      for (const n of this.roots) {
        n.s = clamp(this.semblance(n.c) + 0.06 * gauss(), 0, 1);
        n.delay = rand(0, 0.5);
        n.split = false;
      }
      this.nodes = this.roots.slice();
      this.est = null;
      this.estTarget = null;
      this.locked = false;

      for (const st of this.stations) {
        st.dist = Math.sqrt(dist2(st.p, this.h));
        st.hot = 0;
      }
      const order = this.stations.slice().sort((a, b) => a.dist - b.dist);
      order.forEach(
          (st, i) => { st.rayAt = T.rays + i * 0.07 + rand(0, 0.08); });

      this.queue = SPLITS.map((_, level) => ({
                                at : T.split + level * T.step,
                                run : () => this.refine(level),
                              }));
      this.queue.push(
          {at : T.lock, run : () => (this.locked = true)},
          {at : T.catalog, run : () => this.addToCatalog()},
      );
    }

    refine(level) {
      const at = this.t;
      const candidates =
          this.nodes.filter((n) => n.level === level && !n.pruned)
              .sort((a, b) => b.s - a.s);
      const chosen = candidates.slice(0, SPLITS[level]);
      const home = candidates.find(
          (n) => n.c.every((c, i) => Math.abs(this.h[i] - c) <= n.h + 1e-9),
      );
      if (home && !chosen.includes(home))
        chosen[chosen.length - 1] = home;

      const children = [];
      for (const n of chosen) {
        n.split = true;
        n.splitAt = at;
        const h = n.h / 2;
        for (let i = 0; i < 8; i++) {
          const c = [
            n.c[0] + (i & 1 ? h : -h),
            n.c[1] + (i & 2 ? h : -h),
            n.c[2] + (i & 4 ? h : -h),
          ];
          const noise = (0.05 * gauss()) / (level + 1);
          children.push({
            c,
            h,
            level : level + 1,
            parent : n,
            born : at + rand(0, 0.32),
            s : clamp(this.semblance(c) + noise, 0, 1),
          });
        }
      }
      if (level > 0) {
        for (const n of candidates) {
          if (!n.split) {
            n.pruned = true;
            n.prunedAt = at;
          }
        }
      }
      this.nodes.push(...children);

      let best = children[0];
      for (const c of children)
        if (c.s > best.s)
          best = c;
      this.estTarget = level === LEVELS - 1 ? this.h : best.c;
    }

    addToCatalog() {
      this.catalog.push({p : this.h.slice(), no : this.eventNo});
      while (this.catalog.length > CATALOG_SIZE)
        this.catalog.shift();
    }

    request() {
      if (!this.raf && this.visible) {
        this.last = 0;
        this.raf = requestAnimationFrame(this.loop);
      }
    }

    loop(now) {
      this.raf = 0;
      if (!this.svg.isConnected) {
        this.destroy();
        return;
      }
      const dt = this.last ? Math.min((now - this.last) / 1000, 0.05) : 0;
      this.last = now;
      this.tick(dt);
      this.draw();
      if (this.visible)
        this.raf = requestAnimationFrame(this.loop);
    }

    tick(dt) {
      this.clock += dt;
      this.t += dt;
      while (this.queue.length && this.queue[0].at <= this.t)
        this.queue.shift().run();

      if (this.estTarget) {
        if (!this.est)
          this.est = this.estTarget.slice();
        const u = 1 - Math.exp(-dt * 7);
        this.est = lerp3(this.est, this.estTarget, u);
      }
      if (this.t >= T.end + this.pause)
        this.startEvent();
    }

    destroy() {
      cancelAnimationFrame(this.raf);
      this.resizer?.disconnect();
      this.observer?.disconnect();
    }

    resize() {
      const box = this.host.getBoundingClientRect();
      if (!box.width || !box.height)
        return;
      const W = box.width;
      const H = box.height;
      this.svg.setAttribute("viewBox", `0 0 ${r1(W)} ${r1(H)}`);

      const hero = this.host.parentElement.getBoundingClientRect();
      const left = hero.left - box.left;
      if (hero.width >= 760) {
        this.cx = left + hero.width * 0.77;
        this.cy = H * 0.5;
        this.scale = Math.min(hero.width * 0.2, H * 0.33);
      } else {
        this.cx = W * 0.5;
        this.cy = H * 0.5;
        this.scale = Math.min(W * 0.36, H * 0.3);
      }
      this.host.style.setProperty("--oct-x", `${r1(this.cx)}px`);
      this.host.style.setProperty("--oct-y", `${r1(this.cy)}px`);
      if (this.reduced)
        this.draw();
    }

    camera() {
      const c = this.clock;
      const yaw = this.yaw0 + c * ((2 * Math.PI) / 140) +
                  0.25 * Math.sin(c * 0.023 + this.yawDrift);
      const pitch = 0.4 + 0.07 * Math.sin(c * 0.051 + this.yawDrift);
      this.cyaw = Math.cos(yaw);
      this.syaw = Math.sin(yaw);
      this.cpitch = Math.cos(pitch);
      this.spitch = Math.sin(pitch);
    }

    /** Project a point to screen coordinates: [x, y, depth, perspective]. */
    project(p) {
      const x = p[0] * this.cyaw - p[1] * this.syaw;
      const y = p[0] * this.syaw + p[1] * this.cyaw;
      const depth = y * this.cpitch - p[2] * this.spitch;
      const up = y * this.spitch + p[2] * this.cpitch;
      const k = CAMERA / (CAMERA + depth);
      return [
        this.cx + this.scale * x * k, this.cy - this.scale * up * k, depth, k
      ];
    }

    cube(c, hx, hy, hz) {
      const P = [];
      for (let i = 0; i < 8; i++) {
        P.push(
            this.project([
              c[0] + (i & 1 ? hx : -hx),
              c[1] + (i & 2 ? hy : -hy),
              c[2] + (i & 4 ? hz : -hz),
            ]),
        );
      }
      let d = "";
      for (const [a, b] of EDGES) {
        d += `M${r1(P[a][0])} ${r1(P[a][1])}L${r1(P[b][0])} ${r1(P[b][1])}`;
      }
      return d;
    }

    line(p, q) {
      const a = this.project(p);
      const b = this.project(q);
      return `M${r1(a[0])} ${r1(a[1])}L${r1(b[0])} ${r1(b[1])}`;
    }

    /** Opacity of the search, fading out at the end of an event. */
    get fadeOut() { return 1 - smooth(T.fade, T.fade + 1.1, this.t); }

    draw() {
      if (this.cx === undefined)
        return;
      this.camera();
      const t = this.t;
      const fade = this.fadeOut;

      // Search volume and the surface grid the stations sit on.
      this.frame.setAttribute("d", this.cube([ 0, 0, 0 ], ...HALF));
      let grid = "";
      const top = HALF[2];
      for (let i = 0; i <= ROOT[0]; i++) {
        const u = -1 + (2 * i) / ROOT[0];
        grid += this.line([ u * HALF[0], -HALF[1], top ],
                          [ u * HALF[0], HALF[1], top ]);
        grid += this.line([ -HALF[0], u * HALF[1], top ],
                          [ HALF[0], u * HALF[1], top ]);
      }
      this.surface.setAttribute("d", grid);

      this.drawCatalog();
      this.drawNodes(t, fade);
      this.drawWaves(t);
      this.drawQuake(t);
      this.drawRays(t, fade);
      this.drawMarkers(t, fade);
    }

    drawCatalog() {
      const d = [ "", "", "" ];
      for (const ev of this.catalog) {
        const age = this.eventNo - ev.no;
        const b = age < 3 ? 0 : age < 14 ? 1 : 2;
        const [x, y] = this.project(ev.p);
        d[b] += `M${r1(x)} ${r1(y)}h0`;
      }
      this.catalogPaths.forEach((p, i) => p.setAttribute("d", d[i]));
    }

    drawNodes(t, fade) {
      const dots = this.dotPaths.map((row) => row.map(() => ""));
      const cubes = this.cubePaths.map((row) => row.map(() => ""));
      let pruned = "";

      for (const n of this.nodes) {
        let c = n.c;
        let grow = 1;
        if (n.level > 0) {
          grow = ease((t - n.born) / T.grow);
          // Fold the tree back, finest levels first.
          const fold = smooth(
              T.fade + (LEVELS - n.level) * 0.14,
              T.fade + (LEVELS - n.level) * 0.14 + 0.7,
              t,
          );
          grow *= 1 - fold;
          if (grow <= 0.001)
            continue;
          c = lerp3(n.parent.c, n.c, grow);
          const hh = n.h * grow;
          const [, , depth] = this.project(c);
          const b = clamp(Math.floor(((depth + 1.4) / 2.8) * DEPTH_BUCKETS), 0,
                          DEPTH_BUCKETS - 1);
          const d = this.cube(c, hh, hh, hh);
          if (n.pruned)
            pruned += d;
          else
            cubes[n.level - 1][b] += d;
        }

        // A refined node hands its sample over to its children until the tree
        // folds.
        if (n.split && t > n.splitAt + 0.15 && t < T.fade + 0.9)
          continue;
        let v;
        if (n.level === 0) {
          const stack = smooth(T.stack + n.delay, T.stack + n.delay + 1.1, t);
          v = 0.1 +
              0.9 * n.s * stack * (n.split ? 0.4 : 1) * (0.25 + 0.75 * fade);
        } else {
          v = n.s * smooth(0.3, 1, grow) * (n.pruned ? 0.3 : 1);
        }
        if (v < 0.05)
          continue;
        const b = Math.min(DOT_BUCKETS - 1, Math.floor(v * DOT_BUCKETS));
        const [x, y] = this.project(c);
        dots[n.level][b] += `M${r1(x)} ${r1(y)}h0`;
      }

      this.dotPaths.forEach(
          (row, l) => row.forEach((p, b) => p.setAttribute("d", dots[l][b])));
      this.cubePaths.forEach(
          (row, l) => row.forEach((p, b) => p.setAttribute("d", cubes[l][b])));
      this.prunedPath.setAttribute("d", pruned);
    }

    /** Radius of a wavefront in direction g, bent by the heterogeneities. */
    bend(g, r) {
      let f = 0;
      for (const w of this.wobble) {
        const kg = w.k[0] * g[0] + w.k[1] * g[1] + w.k[2] * g[2];
        f += w.amp * Math.sin(w.freq * kg + w.phase + 0.5 * this.t);
      }
      return r * (1 + f);
    }

    /**
     * Path data of a wavefront of radius r around the source: its outline
     * below the surface and the ring where it reaches the surface.
     */
    wavefront(r) {
      const h = this.h;
      const top = HALF[2];
      let d = "";
      let pen = false;
      const to = (p) => {
        const [x, y] = this.project(p);
        d += `${pen ? "L" : "M"}${r1(x)} ${r1(y)}`;
        pen = true;
      };
      const surface = (a, b) => lerp3(a, b, (top - a[2]) / (b[2] - a[2]));

      // Outline facing the camera, cut off at the surface.
      const R = [ this.cyaw, -this.syaw, 0 ];
      const U =
          [ this.syaw * this.spitch, this.cyaw * this.spitch, this.cpitch ];
      let prev = null;
      for (let i = 0; i <= 96; i++) {
        const a = (i / 96) * 2 * Math.PI;
        const g =
            [ 0, 1, 2 ].map((j) => Math.cos(a) * R[j] + Math.sin(a) * U[j]);
        const rr = this.bend(g, r);
        const p = [ h[0] + rr * g[0], h[1] + rr * g[1], h[2] + rr * g[2] ];
        if (p[2] <= top) {
          if (prev && prev[2] > top)
            to(surface(prev, p));
          to(p);
        } else if (prev && prev[2] <= top) {
          to(surface(prev, p));
          pen = false;
        }
        prev = p;
      }

      // The ring spreading over the surface.
      const below = top - h[2];
      pen = false;
      if (r > below) {
        const rho0 = Math.sqrt(r * r - below * below);
        for (let i = 0; i <= 72; i++) {
          const a = (i / 72) * 2 * Math.PI;
          const g =
              [ (rho0 * Math.cos(a)) / r, (rho0 * Math.sin(a)) / r, below / r ];
          const rr = this.bend(g, r);
          if (rr <= below) {
            pen = false;
            continue;
          }
          const rho = Math.sqrt(rr * rr - below * below);
          to([ h[0] + rho * Math.cos(a), h[1] + rho * Math.sin(a), top ]);
        }
      }
      return d;
    }

    drawWaves(t) {
      for (const [wave, speed] of [[ this.waveP, V_P ], [ this.waveS, V_S ]]) {
        const r = speed * t;
        const o =
            r > 0 ? 0.85 * (1 - smooth(0.3, 2.4, r)) * smooth(0, 0.12, r) : 0;
        wave.g.setAttribute("opacity", o.toFixed(3));
        const fronts = RIPPLES.map((_, i) => {
          const ri = r - i * WAVELENGTH * (1 + 0.6 * r);
          return o > 0.003 && ri > 0 ? this.wavefront(ri) : "";
        });
        wave.crests.forEach((p, i) => p.setAttribute("d", fronts[i]));
        wave.glow.setAttribute("d", fronts[0]);
      }
    }

    /** The earthquake: a warm glow and the rupture spreading over the fault. */
    drawQuake(t) {
      const on = 1 - smooth(1.2, 2.8, t);
      this.quake.setAttribute("opacity", on.toFixed(3));
      if (on <= 0)
        return;
      const h = this.h;
      const {s, d} = this.fault;
      const [x, y, , k] = this.project(h);
      const flash = Math.exp(-t * 1.8);
      this.quakeGlow.setAttribute("cx", r1(x));
      this.quakeGlow.setAttribute("cy", r1(y));
      this.quakeGlow.setAttribute("r", r1((26 + 44 * flash) * k));
      this.quakeCore.setAttribute("cx", r1(x));
      this.quakeCore.setAttribute("cy", r1(y));
      this.quakeCore.setAttribute("r", r1((2.4 + 2.4 * flash) * k));

      // An irregular rupture patch growing on the fault plane.
      const grow = 0.16 * ease(t / 1.4);
      let patch = "";
      for (let i = 0; i < 48; i++) {
        const a = (i / 48) * 2 * Math.PI;
        let rho = 1;
        this.wobble.forEach(
            (w, j) => { rho += 7 * w.amp * Math.sin((j + 2) * a + w.phase); });
        rho *= grow;
        const c = Math.cos(a) * rho;
        const sn = Math.sin(a) * rho;
        const [px, py] =
            this.project([ 0, 1, 2 ].map((j) => h[j] + c * s[j] + sn * d[j]));
        patch += `${i ? "L" : "M"}${r1(px)} ${r1(py)}`;
      }
      this.quakePatch.setAttribute("d", `${patch}Z`);
    }

    drawRays(t, fade) {
      const h = this.h;
      for (const st of this.stations) {
        // P arrival at the station, then the migrated ray back to the source.
        const arrival = st.dist / V_P;
        const ping = t > arrival ? Math.exp(-(t - arrival) * 2.6) : 0;
        const ray = ease((t - st.rayAt) / 0.75);
        const linked = ray >= 1 ? fade : 0;

        if (ray > 0 && fade > 0) {
          let d = "";
          const span = Math.hypot(st.p[0] - h[0], st.p[1] - h[1]);
          for (let i = 0; i <= 16; i++) {
            const u = i / 16;
            const p = lerp3(h, st.p, u);
            p[2] -= 0.16 * span * Math.sin(Math.PI * u) * (1 - u * 0.4);
            const [x, y] = this.project(p);
            d += `${i ? "L" : "M"}${r1(x)} ${r1(y)}`;
          }
          st.ray.setAttribute("d", d);
          st.ray.setAttribute("stroke-dashoffset", (1 - ray).toFixed(3));
          st.ray.setAttribute("stroke-opacity", (0.38 * fade).toFixed(3));
        } else {
          st.ray.setAttribute("d", "");
        }

        const [x, y] = this.project(st.p);
        const s = 5 + 3 * ping + 0.8 * linked;
        st.el.setAttribute(
            "d",
            `M${r1(x)} ${r1(y - s)}L${r1(x + s * 0.87)} ${r1(y + s * 0.5)}L${
                r1(x - s * 0.87)} ${r1(y + s * 0.5)}Z`,
        );
        st.el.setAttribute(
            "fill-opacity",
            (0.35 + 0.45 * Math.max(ping, 0.6 * linked)).toFixed(3));
        st.el.classList.toggle("is-linked", linked > 0.5);
      }
    }

    drawMarkers(t, fade) {
      // Running estimate of the source while the octree refines.
      const lock = smooth(T.lock, T.lock + 0.45, t);
      if (this.est && fade > 0) {
        const [x, y] = this.project(this.est);
        const r = 10 - 6 * lock;
        const g = r + 5;
        this.estimate.setAttribute("opacity", ((1 - lock) * 0.9).toFixed(3));
        this.estRing.setAttribute("cx", r1(x));
        this.estRing.setAttribute("cy", r1(y));
        this.estRing.setAttribute("r", r1(r));
        this.estTicks.setAttribute(
            "d",
            `M${r1(x - g)} ${r1(y)}h${r1(g - r + 2)}M${r1(x + g)} ${r1(y)}h${
                r1(r - g - 2)}` +
                `M${r1(x)} ${r1(y - g)}v${r1(g - r + 2)}M${r1(x)} ${
                    r1(y + g)}v${r1(r - g - 2)}`,
        );
      } else {
        this.estimate.setAttribute("opacity", 0);
      }

      // The located hypocentre with its epicentre projected to the surface.
      const shown = lock * fade;
      this.hypo.setAttribute("opacity", shown.toFixed(3));
      if (shown <= 0) {
        this.drop.setAttribute("d", "");
        return;
      }
      const [x, y, , k] = this.project(this.h);
      const epi = [ this.h[0], this.h[1], HALF[2] ];
      const drop = smooth(T.lock + 0.2, T.lock + 0.9, t);
      this.drop.setAttribute("d", this.line(this.h, lerp3(this.h, epi, drop)));
      this.drop.setAttribute("stroke-opacity", (0.55 * shown).toFixed(3));

      const breathe = 1 + 0.08 * Math.sin(t * 3.1);
      this.hypoGlow.setAttribute("cx", r1(x));
      this.hypoGlow.setAttribute("cy", r1(y));
      this.hypoGlow.setAttribute("r",
                                 r1(34 * k * breathe * (0.6 + 0.4 * lock)));
      this.hypoCore.setAttribute("cx", r1(x));
      this.hypoCore.setAttribute("cy", r1(y));
      this.hypoCore.setAttribute("r", r1(3.3 * k));
      this.pulses.forEach((p, i) => {
        const u = (((t - T.lock) / 1.7 + i / 2) % 1 + 1) % 1;
        p.setAttribute("cx", r1(x));
        p.setAttribute("cy", r1(y));
        p.setAttribute("r", r1((4 + 34 * ease(u)) * k));
        p.setAttribute("stroke-opacity", (0.7 * (1 - u)).toFixed(3));
      });
    }
  }

  let current = null;

  const mount = () => {
    const host = document.querySelector(".qs-hero__bg");
    if (current && current.host === host)
      return;
    current?.destroy();
    current = host ? new OctreeBackdrop(host) : null;
  };

  // Material's instant navigation swaps the page without reloading scripts.
  if (typeof window.document$?.subscribe === "function") {
    window.document$.subscribe(mount);
  } else if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", mount);
  } else {
    mount();
  }
})();

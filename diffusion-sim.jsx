import { useEffect, useRef, useState, useCallback } from "react";
import * as THREE from "three";

const N = 18000;

// ── Noise schedule (linear beta) ────────────────────────────────────────────
function buildSchedule(T = 1000) {
  const alphas_bar = new Float32Array(T);
  let ab = 1.0;
  for (let t = 0; t < T; t++) {
    const beta = 0.0001 + (0.02 - 0.0001) * (t / (T - 1));
    ab *= (1 - beta);
    alphas_bar[t] = ab;
  }
  return alphas_bar;
}

const SCHEDULE = buildSchedule();

// ── Target shapes ─────────────────────────────────────────────────────────────
function sampleShape(name, n) {
  const pos = new Float32Array(n * 3);
  const rng = () => Math.random();
  const gauss = () => {
    const u = rng() + 1e-10, v = rng();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  };

  switch (name) {
    case "torus": {
      for (let i = 0; i < n; i++) {
        const u = rng() * Math.PI * 2, v = rng() * Math.PI * 2;
        const R = 1.8, r = 0.65;
        pos[i*3]   = (R + r * Math.cos(v)) * Math.cos(u);
        pos[i*3+1] = (R + r * Math.cos(v)) * Math.sin(u);
        pos[i*3+2] = r * Math.sin(v);
      }
      break;
    }
    case "sphere": {
      for (let i = 0; i < n; i++) {
        const phi = Math.acos(2 * rng() - 1), th = rng() * Math.PI * 2;
        const r = 2.2 + gauss() * 0.05;
        pos[i*3]   = r * Math.sin(phi) * Math.cos(th);
        pos[i*3+1] = r * Math.sin(phi) * Math.sin(th);
        pos[i*3+2] = r * Math.cos(phi);
      }
      break;
    }
    case "helix": {
      for (let i = 0; i < n; i++) {
        const t = (i / n) * Math.PI * 12;
        const nr = gauss() * 0.12;
        pos[i*3]   = (Math.cos(t) + gauss() * 0.05) * (2 + nr);
        pos[i*3+1] = (Math.sin(t) + gauss() * 0.05) * (2 + nr);
        pos[i*3+2] = (t / (Math.PI * 12)) * 5.5 - 2.75;
      }
      break;
    }
    case "trefoil": {
      for (let i = 0; i < n; i++) {
        const t = (i / n) * Math.PI * 2;
        const s = 1.1;
        pos[i*3]   = s * (Math.sin(t) + 2 * Math.sin(2 * t)) + gauss() * 0.08;
        pos[i*3+1] = s * (Math.cos(t) - 2 * Math.cos(2 * t)) + gauss() * 0.08;
        pos[i*3+2] = s * (-Math.sin(3 * t))                   + gauss() * 0.08;
      }
      break;
    }
    case "möbius": {
      for (let i = 0; i < n; i++) {
        const u = rng() * Math.PI * 2;
        const v = (rng() - 0.5) * 1.4;
        const R = 2;
        pos[i*3]   = (R + v * Math.cos(u / 2)) * Math.cos(u);
        pos[i*3+1] = (R + v * Math.cos(u / 2)) * Math.sin(u);
        pos[i*3+2] = v * Math.sin(u / 2);
      }
      break;
    }
    default: {
      for (let i = 0; i < n * 3; i++) pos[i] = gauss() * 2;
    }
  }
  return pos;
}

// ── GLSL ─────────────────────────────────────────────────────────────────────
const vert = `
  attribute float aTemp;
  attribute float aSize;
  varying float vTemp;
  void main() {
    vTemp = aTemp;
    vec4 mv = modelViewMatrix * vec4(position, 1.0);
    gl_PointSize = aSize * (280.0 / -mv.z);
    gl_Position = projectionMatrix * mv;
  }
`;

const frag = `
  varying float vTemp;
  uniform float uTime;
  void main() {
    vec2 uv = gl_PointCoord - 0.5;
    float d = length(uv);
    if (d > 0.5) discard;
    float alpha = (1.0 - smoothstep(0.15, 0.5, d)) * 0.88;

    // hot orange → violet → ice blue → crystal white
    vec3 hot     = vec3(1.00, 0.52, 0.08);
    vec3 violet  = vec3(0.55, 0.15, 0.95);
    vec3 ice     = vec3(0.15, 0.72, 1.00);
    vec3 crystal = vec3(0.88, 0.96, 1.00);

    vec3 col;
    if (vTemp > 0.66)      col = mix(violet,  hot,     (vTemp - 0.66) / 0.34);
    else if (vTemp > 0.33) col = mix(ice,     violet,  (vTemp - 0.33) / 0.33);
    else                   col = mix(crystal, ice,     vTemp / 0.33);

    col += hot * pow(vTemp, 3.0) * 0.4;

    gl_FragColor = vec4(col, alpha);
  }
`;

// ── Component ─────────────────────────────────────────────────────────────────
const SHAPES = ["torus", "sphere", "helix", "trefoil", "möbius"];

export default function DiffusionSim() {
  const mountRef  = useRef(null);
  const stRef     = useRef({ t: 999, playing: true, speed: 4, shape: "torus", noise: null, target: null, forward: true });
  const gfxRef    = useRef(null);
  const dragRef   = useRef({ active: false, px: 0, py: 0, rotX: 0.25, rotY: 0 });
  const [ui, setUi] = useState({ t: 999, playing: true, speed: 4, shape: "torus", forward: true });

  // ── Initialise / reset particles ──────────────────────────────────────────
  const resetParticles = useCallback((shape) => {
    const st = stRef.current;
    st.target = sampleShape(shape, N);
    const eps = new Float32Array(N * 3);
    for (let i = 0; i < N * 3; i++) {
      const u = Math.random() + 1e-10, v = Math.random();
      eps[i] = Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
    }
    st.noise = eps;
    st.t = 999;
    st.forward = true;
    setUi(p => ({ ...p, t: 999, forward: true }));
  }, []);

  // ── Three.js setup (once) ─────────────────────────────────────────────────
  useEffect(() => {
    const mount = mountRef.current;
    const W = mount.clientWidth, H = mount.clientHeight;

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(W, H);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setClearColor(0x04040a);
    mount.appendChild(renderer.domElement);

    const scene  = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(58, W / H, 0.01, 100);
    camera.position.set(0, 0, 9);

    // geometry
    const geo  = new THREE.BufferGeometry();
    const pos  = new Float32Array(N * 3);
    const temp = new Float32Array(N);
    const sz   = new Float32Array(N);
    geo.setAttribute("position", new THREE.BufferAttribute(pos,  3));
    geo.setAttribute("aTemp",    new THREE.BufferAttribute(temp, 1));
    geo.setAttribute("aSize",    new THREE.BufferAttribute(sz,   1));

    const mat = new THREE.ShaderMaterial({
      vertexShader: vert, fragmentShader: frag,
      uniforms: { uTime: { value: 0 } },
      transparent: true, depthWrite: false,
      blending: THREE.AdditiveBlending,
    });

    const pts = new THREE.Points(geo, mat);
    scene.add(pts);
    gfxRef.current = { geo, mat, pos, temp, sz, pts };

    resetParticles(stRef.current.shape);

    // ── drag rotation ─────────────────────────────────────────────────────
    const onDown = e => { dragRef.current.active = true; dragRef.current.px = e.clientX; dragRef.current.py = e.clientY; };
    const onMove = e => {
      if (!dragRef.current.active) return;
      dragRef.current.rotY += (e.clientX - dragRef.current.px) * 0.006;
      dragRef.current.rotX += (e.clientY - dragRef.current.py) * 0.006;
      dragRef.current.px = e.clientX; dragRef.current.py = e.clientY;
    };
    const onUp = () => { dragRef.current.active = false; };
    mount.addEventListener("mousedown", onDown);
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);

    // ── resize ────────────────────────────────────────────────────────────
    const onResize = () => {
      const w = mount.clientWidth, h = mount.clientHeight;
      camera.aspect = w / h; camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    };
    window.addEventListener("resize", onResize);

    // ── render loop ───────────────────────────────────────────────────────
    let rafId;
    const clock = new THREE.Clock();
    let looping = false;

    const loop = () => {
      rafId = requestAnimationFrame(loop);
      const elapsed = clock.getElapsedTime();
      mat.uniforms.uTime.value = elapsed;

      const st = stRef.current;

      // advance timestep: forward = noise→object (t 999→0), !forward = object→noise (t 0→999)
      if (st.playing && !looping) {
        if (st.forward) {
          st.t = st.t - st.speed;
          if (st.t <= 0) {
            st.t = 0;
            looping = true;
            setTimeout(() => {
              st.forward = false;
              looping = false;
              setUi(p => ({ ...p, t: 0, forward: false }));
            }, 1800);
          } else {
            setUi(p => ({ ...p, t: Math.round(st.t) }));
          }
        } else {
          st.t = st.t + st.speed;
          if (st.t >= 999) {
            st.t = 999;
            looping = true;
            setTimeout(() => {
              // don't resample: current positions are already the noise state → seamless loop
              st.forward = true;
              looping = false;
              setUi(p => ({ ...p, t: 999, forward: true }));
            }, 1800);
          } else {
            setUi(p => ({ ...p, t: Math.round(st.t) }));
          }
        }
      }

      // update positions: x_t = sqrt(ᾱ_t)·x₀ + sqrt(1-ᾱ_t)·ε
      const { target, noise } = st;
      if (target && noise) {
        const tIdx = Math.max(0, Math.min(999, Math.floor(st.t)));
        const sab  = Math.sqrt(SCHEDULE[tIdx]);
        const s1ab = Math.sqrt(1 - SCHEDULE[tIdx]);
        const tNorm = st.t / 999;

        for (let i = 0; i < N; i++) {
          const i3 = i * 3;
          pos[i3]   = sab * target[i3]   + s1ab * noise[i3];
          pos[i3+1] = sab * target[i3+1] + s1ab * noise[i3+1];
          pos[i3+2] = sab * target[i3+2] + s1ab * noise[i3+2];
          temp[i]   = tNorm;
          sz[i]     = 0.9 + tNorm * 1.8 + Math.sin(elapsed * 18 + i * 0.7) * tNorm * 0.35;
        }
        geo.attributes.position.needsUpdate = true;
        geo.attributes.aTemp.needsUpdate    = true;
        geo.attributes.aSize.needsUpdate    = true;
      }

      // rotation
      const drag = dragRef.current;
      pts.rotation.x = drag.rotX;
      pts.rotation.y = drag.rotY + elapsed * 0.04;

      renderer.render(scene, camera);
    };
    loop();

    return () => {
      cancelAnimationFrame(rafId);
      mount.removeEventListener("mousedown", onDown);
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
      window.removeEventListener("resize", onResize);
      renderer.dispose();
      if (mount.contains(renderer.domElement)) mount.removeChild(renderer.domElement);
    };
  }, [resetParticles]);

  const changeShape = (s) => {
    stRef.current.shape = s;
    resetParticles(s);
    setUi(p => ({ ...p, shape: s }));
  };

  const togglePlay = () => {
    stRef.current.playing = !stRef.current.playing;
    setUi(p => ({ ...p, playing: !p.playing }));
  };

  const tPct = (1 - ui.t / 999) * 100;

  const mono = { fontFamily: '"Courier New", monospace' };

  return (
    <div style={{ width: "100vw", height: "100vh", background: "#04040a", position: "relative", overflow: "hidden" }}>

      {/* Canvas */}
      <div ref={mountRef} style={{ width: "100%", height: "100%", cursor: "grab" }} />

      {/* Top-left label */}
      <div style={{ position: "absolute", top: 28, left: 32, ...mono, userSelect: "none" }}>
        <div style={{ color: "rgba(255,255,255,0.75)", fontSize: 15, letterSpacing: "0.25em" }}>DIFFUSION</div>
        <div style={{ color: "rgba(255,255,255,0.2)", fontSize: 9, letterSpacing: "0.4em", marginTop: 3 }}>DENOISING SIMULATION</div>
      </div>

      {/* Top-right equation */}
      <div style={{ position: "absolute", top: 28, right: 32, textAlign: "right", ...mono, userSelect: "none" }}>
        <div style={{ color: "rgba(96,165,250,0.45)", fontSize: 10, letterSpacing: "0.05em" }}>
          x<sub>t</sub> = √ᾱ<sub>t</sub> · x<sub>0</sub> + √(1−ᾱ<sub>t</sub>) · ε
        </div>
        <div style={{ color: "rgba(255,255,255,0.12)", fontSize: 9, letterSpacing: "0.2em", marginTop: 4 }}>
          drag to rotate
        </div>
      </div>

      {/* Bottom controls */}
      <div style={{
        position: "absolute", bottom: 36, left: "50%", transform: "translateX(-50%)",
        display: "flex", flexDirection: "column", alignItems: "center", gap: 14,
        ...mono,
      }}>

        {/* Timestep + phase label */}
        <div style={{ display: "flex", alignItems: "center", gap: 20 }}>
          <div style={{ color: "rgba(96,165,250,0.7)", fontSize: 12, letterSpacing: "0.3em" }}>
            t = {String(Math.round(ui.t)).padStart(4, "0")}
          </div>
          <div style={{
            fontSize: 9, letterSpacing: "0.35em",
            color: ui.t > 666 ? "#ff8c42" : ui.t > 333 ? "#a78bfa" : "#38bdf8",
            transition: "color 0.3s",
          }}>
            {ui.t > 666 ? "DIFFUSING" : ui.t > 333 ? "CONDENSING" : "CRYSTALLISING"}
          </div>
        </div>

        {/* Progress bar */}
        <div style={{ width: 320, height: 2, background: "rgba(255,255,255,0.07)", borderRadius: 1 }}>
          <div style={{
            width: `${tPct}%`, height: "100%", borderRadius: 1,
            background: "linear-gradient(90deg, #ff9500 0%, #7c3aed 50%, #38bdf8 100%)",
            transition: "width 0.06s linear",
          }} />
        </div>

        {/* Shape buttons */}
        <div style={{ display: "flex", gap: 6 }}>
          {SHAPES.map(s => (
            <button key={s} onClick={() => changeShape(s)} style={{
              padding: "5px 13px", cursor: "pointer",
              ...mono, fontSize: 10, letterSpacing: "0.2em", textTransform: "uppercase",
              background: ui.shape === s ? "rgba(96,165,250,0.12)" : "transparent",
              border: `1px solid ${ui.shape === s ? "rgba(96,165,250,0.6)" : "rgba(255,255,255,0.12)"}`,
              color: ui.shape === s ? "#60a5fa" : "rgba(255,255,255,0.3)",
              transition: "all 0.15s",
            }}>{s}</button>
          ))}
        </div>

        {/* Play / speed */}
        <div style={{ display: "flex", gap: 16, alignItems: "center" }}>
          <button onClick={togglePlay} style={{
            ...mono, fontSize: 10, letterSpacing: "0.3em", cursor: "pointer",
            padding: "5px 18px",
            background: "transparent",
            border: "1px solid rgba(255,255,255,0.18)",
            color: "rgba(255,255,255,0.5)",
          }}>
            {ui.playing ? "⏸ PAUSE" : "▶ PLAY"}
          </button>

          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <span style={{ color: "rgba(255,255,255,0.2)", fontSize: 9, letterSpacing: "0.2em" }}>SPEED</span>
            <input type="range" min={1} max={12} value={ui.speed}
              onChange={e => {
                const v = Number(e.target.value);
                stRef.current.speed = v;
                setUi(p => ({ ...p, speed: v }));
              }}
              style={{ width: 90, accentColor: "#60a5fa", cursor: "pointer" }}
            />
            <span style={{ color: "rgba(255,255,255,0.3)", fontSize: 10, ...mono, width: 24 }}>{ui.speed}×</span>
          </div>
        </div>
      </div>

      {/* Heat legend */}
      <div style={{
        position: "absolute", right: 32, top: "50%", transform: "translateY(-50%)",
        display: "flex", flexDirection: "column", alignItems: "center", gap: 6, ...mono,
      }}>
        <div style={{ fontSize: 8, color: "rgba(255,255,255,0.2)", letterSpacing: "0.2em", marginBottom: 4 }}>TEMP</div>
        <div style={{
          width: 4, height: 120, borderRadius: 2,
          background: "linear-gradient(to bottom, #ff9500, #7c3aed, #38bdf8, #e0f0ff)",
        }} />
        <div style={{ fontSize: 8, color: "rgba(255,150,60,0.6)", letterSpacing: "0.1em" }}>HOT</div>
        <div style={{ fontSize: 8, color: "rgba(56,189,248,0.6)", letterSpacing: "0.1em", marginTop: 2 }}>COLD</div>
      </div>

    </div>
  );
}

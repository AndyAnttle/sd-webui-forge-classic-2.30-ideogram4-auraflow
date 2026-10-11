// Ideogram 4 Studio — bbox editor faithful to ComfyUI's ideogram-helper studio.
// Data channel: one hidden textbox (#id4s_state_t2i|i2i) carrying studio JSON.
// Everything else (canvas, element list, inline editor, undo, groups) lives here.
(function () {
    "use strict";

    const MIN_SPAN = 10;          // min group dimension when resizing selection (0..1000)
    const OBJ_COLOR = "#3b82f6";  // matches Comfy studio defaults
    const TEXT_COLOR = "#e0b34c";
    const GROUP_COLORS = ["#f87171", "#34d399", "#60a5fa", "#fbbf24", "#c084fc", "#fb7185"];

    const q = (sel, root) => (root || document).querySelector(sel);
    const qAll = (sel, root) => Array.from((root || document).querySelectorAll(sel));
    const clamp = (v) => Math.max(0, Math.min(1000, Math.round(v)));
    const esc = (s) => String(s ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
    const hexRgba = (hex, a) => {
        const h = (hex || "#000000").replace("#", "");
        const r = parseInt(h.slice(0, 2), 16) || 0, g = parseInt(h.slice(2, 4), 16) || 0, b = parseInt(h.slice(4, 6), 16) || 0;
        return `rgba(${r},${g},${b},${a})`;
    };

    function setStudioState(tab, obj) {
        const tb = q(`#id4s_state_${tab} textarea`) || q(`#id4s_state_${tab} input`);
        if (!tb) return false;
        tb.value = JSON.stringify(obj);
        tb.dispatchEvent(new Event("input", { bubbles: true }));
        return true;
    }

    // generation size inputs (txt2img sliders) - read and optionally write.
    // Gradio 4 inputs are bound via framework events: use the native value
    // setter so the framework sees the change.
    function genSize(tab) {
        const ids = tab === "i2i" ? ["#img2img_width", "#img2img_height"] : ["#txt2img_width", "#txt2img_height"];
        const get = (id) => { const el = q(id + " input") || q(id); const v = el ? parseInt(el.value) : 0; return v > 0 ? v : 0; };
        return { w: get(ids[0]) || 1024, h: get(ids[1]) || 1024, ids };
    }
    function setGenSize(tab, w, h) {
        const ids = tab === "i2i" ? ["#img2img_width", "#img2img_height"] : ["#txt2img_width", "#txt2img_height"];
        const nativeSetter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, "value")?.set;
        const put = (id, v) => {
            const el = q(id + " input") || q(id);
            if (!el) return;
            if (nativeSetter) nativeSetter.call(el, String(v)); else el.value = String(v);
            el.dispatchEvent(new Event("input", { bubbles: true }));
            el.dispatchEvent(new Event("change", { bubbles: true }));
        };
        put(ids[0], w); put(ids[1], h);
    }

    const snap = (v, div) => Math.max(16, Math.round(v / div) * div);

    // write a value into a tab's native numeric input (steps / cfg_scale)
    function setTabField(tab, name, value) {
        const el = q(`#${tab === "i2i" ? "img2img" : "txt2img"}_${name} input`)
                || q(`#${tab === "i2i" ? "img2img" : "txt2img"}_${name} ./slider input`)
                || q(`#${tab === "i2i" ? "img2img" : "txt2img"}_${name}`);
        if (!el) return false;
        const nativeSetter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, "value")?.set;
        if (nativeSetter) nativeSetter.call(el, String(value)); else el.value = String(value);
        el.dispatchEvent(new Event("input", { bubbles: true }));
        el.dispatchEvent(new Event("change", { bubbles: true }));
        return true;
    }
    // set a native Gradio Dropdown by elem_id. Gradio 4 (Svelte) dropdowns
    // filter their option list on input; the reliable path is: type the option
    // text, wait for the filtered list to render, then CLICK the exact option.
    function setTabDropdown(tab, name, value, cb) {
        const block = q(`#${tab === "i2i" ? "img2img" : "txt2img"}_${name}`);
        const inp = block && (block.querySelector("input") || block.querySelector("textarea"));
        if (!inp) { if (cb) cb(false); return; }
        const setter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, "value")?.set
                || Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, "value")?.set;
        inp.focus();
        inp.dispatchEvent(new Event("focus", { bubbles: true }));
        if (setter) setter.call(inp, String(value)); else inp.value = String(value);
        inp.dispatchEvent(new Event("input", { bubbles: true }));
        const attempt = (left) => {
            const opts = qAll('[role="option"]', block);
            let target = null;
            for (const o of opts) {
                const t = (o.textContent || "").trim();
                if (t === String(value) || (t.toLowerCase() === String(value).toLowerCase() && o.innerText.trim() === value)) { target = o; break; }
                if (!target && t.toLowerCase().includes(String(value).toLowerCase())) target = o;
            }
            if (target) {
                ["pointerdown", "mousedown", "pointerup", "mouseup", "click"].forEach(t =>
                    target.dispatchEvent(new MouseEvent(t, { bubbles: true, cancelable: true })));
                inp.blur();
                if (cb) cb(true);
            } else if (left > 0) {
                setTimeout(() => attempt(left - 1), 200);
            } else {
                inp.blur();
                if (cb) cb(false);
            }
        };
        setTimeout(() => attempt(8), 150);
    }
    const ASPECTS = [
        ["1:1", 1], ["4:3", 4 / 3], ["3:4", 3 / 4], ["16:9", 16 / 9], ["9:16", 9 / 16],
        ["3:2", 3 / 2], ["2:3", 2 / 3], ["21:9", 21 / 9],
    ];

    function bootStudio(tab) {
        const root = q(`#id4s_canvas_${tab}`);
        if (!root || root.querySelector(":scope > .id4s")) return;
        const LS_KEY = "id4s_state_" + tab;

        // ------------------------------------------------------------------ state
        let state = {
            high_level_description: "", background: "",
            style: {}, // aesthetics / lighting / medium / photo / art_style
            elements: [], // {id,type,bbox|null,desc,text,color_palette,boxColor,groupId,linkId,enabled}
            overlay: { showLabels: true },
        };
        try {
            const saved = JSON.parse(localStorage.getItem(LS_KEY) || "null");
            if (saved && Array.isArray(saved.elements)) state = Object.assign(state, saved);
        } catch { }

        let nextId = 1;
        const reseed = () => { for (const e of state.elements) { const n = parseInt(e.id.slice(1)); if (n >= nextId) nextId = n + 1; } };
        reseed();
        const nid = () => "e" + (nextId++);
        let selectedIds = [];
        const primaryId = () => selectedIds[selectedIds.length - 1] || null;
        const getElement = (id) => state.elements.find(e => e.id === id);
        const isSelected = (id) => selectedIds.includes(id);

        function groupColor(gid) {
            let h = 0; for (const ch of gid) h = (h * 31 + ch.charCodeAt(0)) | 0;
            return GROUP_COLORS[Math.abs(h) % GROUP_COLORS.length];
        }
        const newGroupId = () => "g" + Math.random().toString(36).slice(2, 8);

        // ------------------------------------------------------------------ undo
        const history = [], future = [];
        let lastSnapshot = null;
        const snapshotStr = () => JSON.stringify(state);
        function commit() {
            const s = snapshotStr();
            if (s === lastSnapshot) return;
            if (lastSnapshot !== null) history.push(lastSnapshot);
            if (history.length > 80) history.shift();
            lastSnapshot = s; future.length = 0;
            doPush(); // persist + sync studio JSON to the hidden textbox
        }
        function undo() {
            if (!history.length) return;
            future.push(snapshotStr());
            state = JSON.parse(history.pop());
            lastSnapshot = snapshotStr();
            afterMutate(true);
        }
        function redo() {
            if (!future.length) return;
            history.push(snapshotStr());
            state = JSON.parse(future.pop());
            lastSnapshot = snapshotStr();
            afterMutate(true);
        }

        // ------------------------------------------------------------------ output
        let pushTimer = null;
        function push() {
            clearTimeout(pushTimer);
            pushTimer = setTimeout(doPush, 150);
        }
        function doPush() {
            const out = {
                high_level_description: state.high_level_description,
                background: state.background,
                style: {},
                elements: state.elements
                    .filter(e => e.enabled !== false)
                    .map(e => {
                        const o = { type: e.type };
                        if (e.bbox) o.bbox = e.bbox.map(v => Math.round(v));
                        if (e.type === "text" && e.text) o.text = e.text;
                        if (e.desc) o.desc = e.desc;
                        if (e.color_palette && e.color_palette.length) o.color_palette = e.color_palette.slice(0, 5);
                        return o;
                    }),
            };
            for (const k of ["aesthetics", "lighting", "medium", "photo", "art_style"])
                if (state.style[k]) out.style[k] = state.style[k];
            setStudioState(tab, out);
            persist();
        }
        function persist() {
            try { localStorage.setItem(LS_KEY, JSON.stringify(state)); } catch { }
        }

        function afterMutate(full) {
            renderBoxes(); renderList(); renderEditor(); updateCounts();
            if (full) { /* selection kept */ }
            push();
        }

        // ------------------------------------------------------------------ layout
        root.innerHTML = `
<div class="id4s">
  <div class="id4s-toolbar" id="tb">
    <button class="id4s-ic" data-act="undo" title="Undo (Ctrl+Z)">&#8630;</button>
    <button class="id4s-ic" data-act="redo" title="Redo (Ctrl+Shift+Z)">&#8631;</button>
    <span class="id4s-div"></span>
    <button class="id4s-ic on" data-act="newtype" id="nt-btn" title="New boxes: object — click to switch type · drag on canvas draws a box">▢ obj</button>
    <span class="id4s-div"></span>
    <button class="id4s-ic" data-act="ref" title="Reference image (trace): load file or drop one on canvas">&#128444;</button>
    <label class="id4s-lbl" title="reference opacity">ref <input type="range" min="0" max="1" step="0.05" id="ref-op" style="width:64px" value="0.6"></label>
    <button class="id4s-ic on" data-act="labels" id="labels-btn" title="Show element labels">&#127991;</button>
    <button class="id4s-ic" data-act="fliph" title="Mirror scene horizontally">&#8596;</button>
    <button class="id4s-ic" data-act="flipv" title="Mirror scene vertically">&#8597;</button>
    <span class="id4s-spacer"></span>
    <button class="id4s-txt" data-act="preset-official" title="Official V4_DEFAULT_20: Euler / Simple / 20 / CFG 7 (engine polishes last steps to 3), logit-normal mu=0.0 std=1.75">official</button>
    <button class="id4s-txt" data-act="preset-comfy" title="Comfy ideogram-helper studio: Res Multistep / Simple / 20 / CFG 3, logit-normal mu=0.5 std=1.75">comfy studio</button>
    <button class="id4s-txt" data-act="res" id="res-btn">1024×1024 &#9662;</button>
    <span class="id4s-count" id="cnt"></span>
    <input type="file" accept="image/*" id="ref-file" hidden>
    <div class="id4s-menu" id="res-menu" hidden>
      <div class="id4s-mgroup">aspect ratio</div>
      <div class="id4s-aspects" id="asp-chips"></div>
      <div class="id4s-row"><span>W <input type="number" min="16" max="8192" step="8" id="wh-w" style="width:64px"></span>
        <button class="id4s-ic" data-act="wh-swap" title="Swap width and height">&#8646;</button>
        <span>H <input type="number" min="16" max="8192" step="8" id="wh-h" style="width:64px"></span></div>
      <div class="id4s-hint">size writes into the generation width/height; canvas follows the aspect</div>
    </div>
  </div>

  <div class="id4s-main">
    <div class="id4s-stage-col">
      <div class="id4s-stage" id="stage" tabindex="0">
        <img class="id4s-backdrop" id="backdrop" hidden draggable="false" alt="reference">
        <div class="id4s-grid"></div>
        <div id="boxes"></div>
      </div>
      <div class="id4s-hint" id="hintbar">Drag on canvas — draw a box · Shift+drag — multi-select · double-click a box — type its prompt · Ctrl+C/V copy/paste · Ctrl+G group · Delete remove</div>
    </div>

    <div class="id4s-side">
      <div class="id4s-card">
        <div class="id4s-cardhead">Scene</div>
        <div class="id4s-field"><span>overall</span><textarea data-f="hld" rows="2" placeholder="high-level description of the whole image"></textarea></div>
        <div class="id4s-field"><span>background</span><textarea data-f="bg" rows="2" placeholder="background description"></textarea></div>
        <details class="id4s-style"><summary>style</summary>
          <div class="id4s-field"><span>aesthetics</span><input data-style="aesthetics"></div>
          <div class="id4s-field"><span>lighting</span><input data-style="lighting"></div>
          <div class="id4s-field"><span>medium</span><input data-style="medium" placeholder="photograph / oil painting / …"></div>
          <div class="id4s-field"><span>photo</span><input data-style="photo" placeholder="35mm, high resolution…"></div>
          <div class="id4s-field"><span>art_style</span><input data-style="art_style"></div>
        </details>
      </div>

      <div class="id4s-card id4s-elcard">
        <div class="id4s-cardhead">Elements (<span id="el-count">0</span>)
          <span class="id4s-spacer"></span>
          <button class="id4s-tiny" data-act="add-obj">+ obj</button>
          <button class="id4s-tiny" data-act="add-text">+ text</button>
          <select id="el-filter" title="filter"><option value="all">all</option><option value="obj">objects</option><option value="text">text</option></select>
        </div>
        <ul class="id4s-list" id="ellist"></ul>
      </div>

      <div class="id4s-card" id="edcard" hidden>
        <div class="id4s-cardhead" id="ed-head"></div>
        <div id="ed-body"></div>
      </div>
    </div>
  </div>
</div>`;

        const stage = q("#stage", root), boxesHost = q("#boxes", root);
        const backdrop = q("#backdrop", root);
        let newType = "obj", showLabels = true;

        // ------------------------------------------------------------------ element ops
        function addElement(type, bbox) {
            const e = {
                id: nid(), type, bbox: bbox || [300, 300, 700, 700], desc: "", text: "",
                color_palette: [], boxColor: type === "text" ? TEXT_COLOR : OBJ_COLOR,
                groupId: null, linkId: null, enabled: true,
            };
            state.elements.push(e);
            selectedIds = [e.id];
            commit(); afterMutate();
            return e;
        }
        function removeElements(ids) {
            state.elements = state.elements.filter(e => !ids.includes(e.id));
            selectedIds = selectedIds.filter(id => !ids.includes(id));
            commit(); afterMutate();
        }
        function setShared(id, key, value) {
            const el = getElement(id); if (!el) return;
            const targets = (key !== "boxColor" && key !== "bbox" && key !== "type" && el.linkId)
                ? state.elements.filter(x => x.linkId === el.linkId) : [el];
            for (const t of targets) {
                t[key] = value;
                if (key === "type") t.boxColor = value === "text" ? TEXT_COLOR : (t.boxColor === TEXT_COLOR ? OBJ_COLOR : t.boxColor);
            }
            push(); renderBoxes(); renderList();
        }
        function select(id, additive) {
            if (id === null) { selectedIds = []; }
            else if (additive) { selectedIds = selectedIds.includes(id) ? selectedIds.filter(x => x !== id) : [...selectedIds, id]; }
            else selectedIds = [id];
            renderBoxes(); renderList(); renderEditor();
        }
        function setSelection(ids, additive) {
            if (additive) selectedIds = [...new Set([...selectedIds, ...ids])];
            else selectedIds = ids;
            renderBoxes(); renderList(); renderEditor();
        }
        function boxesUnder(x, y) {
            // topmost = smallest containing box (tightest hit), like real
            // hit-testing in editors; ties go to the later-drawn (on top)
            return state.elements
                .map((e, i) => ({ e, i }))
                .filter(({ e }) => e.bbox && e.enabled !== false &&
                    x >= e.bbox[1] && x <= e.bbox[3] && y >= e.bbox[0] && y <= e.bbox[2])
                .sort((a, b) => {
                    const sa = (a.e.bbox[2] - a.e.bbox[1]) * (a.e.bbox[3] - a.e.bbox[0]);
                    const sb = (b.e.bbox[2] - b.e.bbox[1]) * (b.e.bbox[3] - b.e.bbox[0]);
                    return sa - sb || a.i - b.i;
                })
                .map(({ e }) => e.id);
        }
        function selectedBoxes() {
            return selectedIds.map(getElement).filter(e => e && e.bbox);
        }
        function groupBox() {
            const els = selectedBoxes(); if (els.length < 2) return null;
            let y0 = 1e9, x0 = 1e9, y1 = -1e9, x1 = -1e9;
            for (const e of els) { y0 = Math.min(y0, e.bbox[0]); x0 = Math.min(x0, e.bbox[1]); y1 = Math.max(y1, e.bbox[2]); x1 = Math.max(x1, e.bbox[3]); }
            return [y0, x0, y1, x1];
        }
        let clip = [];
        function copySel() {
            clip = selectedBoxes().map(e => JSON.parse(JSON.stringify(e)));
        }
        function pasteAt(x, y) {
            if (!clip.length) return;
            const gb = (() => {
                let y0 = 1e9, x0 = 1e9, y1 = -1e9, x1 = -1e9;
                for (const c of clip) { y0 = Math.min(y0, c.bbox[0]); x0 = Math.min(x0, c.bbox[1]); y1 = Math.max(y1, c.bbox[2]); x1 = Math.max(x1, c.bbox[3]); }
                return [y0, x0, y1, x1];
            })();
            const dx = x - (gb[1] + gb[3]) / 2, dy = y - (gb[0] + gb[2]) / 2;
            const links = {};
            const copies = clip.map(c => {
                const o = JSON.parse(JSON.stringify(c));
                o.id = nid();
                o.bbox = [clamp(o.bbox[0] + dy), clamp(o.bbox[1] + dx), clamp(o.bbox[2] + dy), clamp(o.bbox[3] + dx)];
                if (o.groupId) { links[o.groupId] = links[o.groupId] || newGroupId(); o.groupId = links[o.groupId]; }
                if (o.linkId) { links[o.linkId] = links[o.linkId] || "l" + Math.random().toString(36).slice(2, 8); o.linkId = links[o.linkId]; }
                state.elements.push(o);
                return o.id;
            });
            selectedIds = copies;
            commit(); afterMutate();
        }
        function linkDuplicate(id) {
            const e = getElement(id); if (!e) return;
            const c = JSON.parse(JSON.stringify(e));
            c.id = nid(); c.groupId = null;
            c.linkId = e.linkId || "l" + Math.random().toString(36).slice(2, 8);
            e.linkId = c.linkId;
            c.bbox = c.bbox ? c.bbox.map(v => clamp(v + (Math.random() * 120 - 60))) : null;
            state.elements.push(c);
            selectedIds = [c.id];
            commit(); afterMutate();
        }
        function unlink(id) {
            const e = getElement(id); if (!e) return;
            e.linkId = null; push(); renderList(); renderEditor();
        }
        function groupSel() {
            if (selectedIds.length < 2) return;
            const gid = newGroupId();
            for (const id of selectedIds) { const e = getElement(id); if (e) e.groupId = gid; }
            commit(); afterMutate();
        }
        function ungroup(id) {
            const e = getElement(id); if (!e || !e.groupId) return;
            for (const x of state.elements) if (x.groupId === e.groupId) x.groupId = null;
            commit(); afterMutate();
        }
        function flipAll(dir) {
            for (const e of state.elements) {
                if (!e.bbox) continue;
                let [y0, x0, y1, x1] = e.bbox;
                if (dir === "h") e.bbox = [y0, 1000 - x1, y1, 1000 - x0];
                else e.bbox = [1000 - y1, x0, 1000 - y0, x1];
            }
            commit(); afterMutate();
        }
        function moveElementTo(id, toIndex) {
            const from = state.elements.findIndex(e => e.id === id);
            if (from === -1 || from === toIndex) return;
            const [e] = state.elements.splice(from, 1);
            state.elements.splice(toIndex, 0, e);
            commit(); afterMutate();
        }

        // ------------------------------------------------------------------ canvas render
        let resW = 1024, resH = 1024;
        function syncRes() {
            stage.style.aspectRatio = `${resW} / ${resH}`;
            q("#res-btn", root).textContent = `${resW}×${resH} ▾`;
            for (const [id, v] of [["#wh-w", resW], ["#wh-h", resH]]) {
                const el = q(id, root);
                if (document.activeElement !== el) el.value = v;  // don't fight the user's typing
            }
            markAspectChip();
            state.res = { w: resW, h: resH };
            persist();
        }
        function markAspectChip() {
            const want = (resW / resH).toFixed(3);
            for (const b of aspHost.querySelectorAll(".id4s-achip"))
                b.classList.toggle("on", (parseFloat(b.dataset.ratio) || 0).toFixed(3) === want);
        }

        function renderBoxes() {
            boxesHost.innerHTML = "";
            const labelOn = showLabels;
            state.elements.forEach((el, i) => {
                if (!el.bbox) return;
                const [y0, x0, y1, x1] = el.bbox;
                const color = el.boxColor || (el.type === "text" ? TEXT_COLOR : OBJ_COLOR);
                const d = document.createElement("div");
                d.className = "id4s-box" + (isSelected(el.id) ? " sel" : "") + (el.id === primaryId() ? " primary" : "") + (el.enabled === false ? " muted" : "");
                d.style.cssText = `left:${x0 / 10}%;top:${y0 / 10}%;width:${(x1 - x0) / 10}%;height:${(y1 - y0) / 10}%;--c:${color};--fill:${hexRgba(color, .16)}`;
                d.dataset.box = el.id;
                if (labelOn) {
                    const tag = document.createElement("span");
                    tag.className = "id4s-tag" + (el.groupId ? " grouped" : "");
                    tag.style.background = hexRgba(color, .88);
                    const gseg = el.groupId ? `<span class="gseg" style="background:${groupColor(el.groupId)}"></span>` : "";
                    const snippet = el.type === "text" ? (el.text || "text") : (el.desc || "obj");
                    tag.innerHTML = `${gseg}<b class="tagn">${i + 1}</b><span class="tagt"> ${esc(snippet).slice(0, 40)}</span>`;
                    d.appendChild(tag);
                }
                if (el.id === primaryId() && selectedIds.length === 1) {
                    for (const h of ["nw", "n", "ne", "e", "se", "s", "sw", "w"]) {
                        const hd = document.createElement("i"); hd.className = "id4s-h h-" + h; hd.dataset.handle = h;
                        d.appendChild(hd);
                    }
                }
                boxesHost.appendChild(d);
            });
            const gb = groupBox();
            if (gb) {
                const els = selectedBoxes();
                const gid = els[0] && els[0].groupId;
                const gc = gid && els.every(e => e.groupId === gid) ? groupColor(gid) : "";
                const [y0, x0, y1, x1] = gb;
                const gf = document.createElement("div");
                gf.className = "id4s-groupframe";
                gf.style.cssText = `left:${x0 / 10}%;top:${y0 / 10}%;width:${(x1 - x0) / 10}%;height:${(y1 - y0) / 10}%;${gc ? `--gc:${gc};` : ""}`;
                for (const h of ["nw", "n", "ne", "e", "se", "s", "sw", "w"]) {
                    const hd = document.createElement("i"); hd.className = "id4s-h h-" + h; hd.dataset.handle = h; hd.dataset.group = "1";
                    gf.appendChild(hd);
                }
                boxesHost.appendChild(gf);
            }
        }

        function updateCounts() {
            q("#cnt", root).textContent = `${state.elements.filter(e => e.enabled !== false).length} active`;
            q("#el-count", root).textContent = String(state.elements.length);
        }

        // ------------------------------------------------------------------ list render
        let filterMode = "all";
        function renderList() {
            const ul = q("#ellist", root);
            ul.innerHTML = "";
            let n = -1;
            for (const el of state.elements) {
                n++;
                if (filterMode !== "all" && el.type !== filterMode) continue;
                const li = document.createElement("li");
                li.className = (isSelected(el.id) ? "sel" : "") + (el.id === primaryId() ? " primary" : "") +
                    (el.groupId ? " grouped" : "") + (el.enabled === false ? " muted" : "");
                if (el.groupId) li.style.setProperty("--gc", groupColor(el.groupId));
                li.dataset.listid = el.id;
                li.dataset.idx = n;
                const color = el.boxColor || (el.type === "text" ? TEXT_COLOR : OBJ_COLOR);
                const snip = el.type === "text" ? "“" + (el.text || "…") + "”" : (el.desc || "object…");
                li.innerHTML =
                    `<i class="id4s-drag" title="drag to reorder">&#8942;&#8942;</i>` +
                    `<button class="id4s-typebtn${el.enabled === false ? " off" : ""}" title="${el.enabled === false ? "muted — click to unmute" : "click to mute (kept, excluded from output)"}">${el.enabled === false ? "◌" : (el.type === "text" ? "T" : "▢")}</button>` +
                    `<span class="id4s-idx" style="background:${color}">${n + 1}</span>` +
                    (el.linkId ? `<i class="id4s-link" title="linked ×${state.elements.filter(x => x.linkId === el.linkId).length}">&#128279;</i>` : "") +
                    `<span class="id4s-snip">${esc(snip).slice(0, 48)}</span>`;
                ul.appendChild(li);
            }
            if (!ul.children.length) {
                const li = document.createElement("li"); li.className = "id4s-empty";
                li.textContent = state.elements.length ? "no elements for this filter" : "No elements yet — Ctrl+drag on the canvas, or + obj / + text.";
                ul.appendChild(li);
            }
        }

        // list interactions
        const ul = q("#ellist", root);
        ul.addEventListener("click", (ev) => {
            const li = ev.target.closest("li[data-listid]"); if (!li) return;
            const id = li.dataset.listid;
            if (ev.target.closest(".id4s-typebtn")) {
                const el = getElement(id); if (el) { el.enabled = el.enabled === false; commit(); afterMutate(); }
                return;
            }
            if (ev.shiftKey || ev.ctrlKey || ev.metaKey) { select(id, true); return; }
            const el = getElement(id);
            if (el && el.groupId) { if (primaryId() === id) select(null); else select(id); return; }
            if (selectedIds.length === 1 && isSelected(id)) select(null); else select(id);
        });
        let dragListId = null;
        ul.addEventListener("dragstart", (ev) => {
            const li = ev.target.closest("li[data-listid]");
            if (!li) return;
            dragListId = li.dataset.listid;
            ev.dataTransfer.effectAllowed = "move";
            li.classList.add("dragging");
        });
        ul.addEventListener("dragover", (ev) => { ev.preventDefault(); });
        ul.addEventListener("drop", (ev) => {
            ev.preventDefault();
            const li = ev.target.closest("li[data-listid]");
            if (!dragListId) return;
            const idx = li ? parseInt(li.dataset.idx) : state.elements.length - 1;
            moveElementTo(dragListId, idx);
            dragListId = null;
        });
        ul.addEventListener("dragend", () => { dragListId = null; renderList(); });
        // make rows draggable only when grabbing the handle
        ul.addEventListener("pointerdown", (ev) => {
            const li = ev.target.closest("li[data-listid]");
            if (li) li.draggable = !!ev.target.closest(".id4s-drag");
        });

        // ------------------------------------------------------------------ element editor
        function renderEditor() {
            const card = q("#edcard", root), head = q("#ed-head", root), body = q("#ed-body", root);
            const el = getElement(primaryId());
            if (!el) { card.hidden = true; return; }
            card.hidden = false;
            const idx = state.elements.indexOf(el);
            const color = el.boxColor || (el.type === "text" ? TEXT_COLOR : OBJ_COLOR);
            head.innerHTML =
                `<label class="id4s-chipcolor" title="element ${idx + 1}${el.linkId ? " · linked ×" + state.elements.filter(x => x.linkId === el.linkId).length : ""} — click to recolour">` +
                `<span class="id4s-idx" style="background:${color}">${idx + 1}</span><input type="color" data-ed="boxColor" value="${color}"></label>` +
                `<button class="id4s-seg${el.type === "obj" ? " on" : ""}" data-ed="type-obj" title="object">▢</button>` +
                `<button class="id4s-seg${el.type === "text" ? " on" : ""}" data-ed="type-text" title="literal text">T</button>` +
                `<button class="id4s-seg${el.bbox ? " on" : ""}" data-ed="togglebbox" title="${el.bbox ? "has a box — click to let the model auto-place" : "auto-placed — click to give it a box"}">${el.bbox ? "▣" : "❔"}</button>` +
                `<span class="id4s-spacer"></span>` +
                `<button class="id4s-seg" data-ed="duplink" title="linked copy — shares the prompt, own position">&#128279;+</button>` +
                (el.linkId ? `<button class="id4s-seg" data-ed="unlink" title="unlink">&#128279;&#822;</button>` : "") +
                `<button class="id4s-seg${el.enabled === false ? " off" : ""}" data-ed="mute" title="${el.enabled === false ? "muted — click to enable" : "mute (keep, exclude from output)"}">${el.enabled === false ? "◌" : "◉"}</button>`;
            let html = "";
            if (el.type === "text")
                html += `<div class="id4s-field"><span>text</span><input data-ed="text" value="${esc(el.text)}" placeholder="literal text the model should draw"></div>`;
            html += `<div class="id4s-field"><span>desc</span><textarea data-ed="desc" rows="3" placeholder="${el.type === "text" ? "how the text looks (font, colour, placement)" : "detailed description of this object"}">${esc(el.desc)}</textarea></div>`;
            html += `<div class="id4s-field"><span>colours ${((el.color_palette || []).length)}/5</span><span class="id4s-pal">${(el.color_palette || []).slice(0, 5).map((c, i) => `<span class="id4s-pchip" style="background:${esc(c)}" data-ed="rmpal" data-pi="${i}" title="click to remove"></span>`).join("")}<input type="color" data-ed="addpal" title="add color (max 5 total)"><button class="id4s-seg" data-ed="copybox" title="replace this element's palette with its canvas marker color (what the model receives)">⬐ canvas</button></span></div>`;
            body.innerHTML = html;
        }
        root.addEventListener("input", (ev) => {
            const ed = ev.target.dataset.ed;
            if (!ed) return;
            const el = getElement(primaryId()); if (!el) return;
            if (ed === "boxColor") setShared(el.id, "boxColor", ev.target.value);
            else if (ed === "text") setShared(el.id, "text", ev.target.value);
            else if (ed === "desc") setShared(el.id, "desc", ev.target.value);
            else if (ed === "addpal" && (ev.target.tagName === "INPUT")) {
                el.color_palette = el.color_palette || [];
                if (el.color_palette.length < 5 && !el.color_palette.includes(ev.target.value)) {
                    el.color_palette.push(ev.target.value); push(); renderEditor(); renderBoxes();
                }
            }
        });
        root.addEventListener("change", (ev) => {
            if (ev.target.dataset.ed === "desc" || ev.target.dataset.ed === "text") commit();
        });
        root.addEventListener("click", (ev) => {
            const ed = ev.target.dataset.ed;
            const el = getElement(primaryId());
            if (ev.target.dataset.ed === "rmpal" && el) {
                el.color_palette.splice(parseInt(ev.target.dataset.pi), 1);
                push(); renderEditor(); renderBoxes(); return;
            }
            if (!ed || !el) return;
            if (ed === "copybox") {
                // replace semantics: the palette becomes exactly this marker's color
                el.color_palette = [(el.boxColor || (el.type === "text" ? TEXT_COLOR : OBJ_COLOR)).toLowerCase()];
                commit(); renderEditor(); renderBoxes();
                return;
            }
            if (ed === "type-obj") { setShared(el.id, "type", "obj"); renderEditor(); }
            else if (ed === "type-text") { setShared(el.id, "type", "text"); renderEditor(); }
            else if (ed === "togglebbox") {
                el.bbox = el.bbox ? null : [300, 300, 700, 700];
                commit(); afterMutate();
            }
            else if (ed === "mute") { el.enabled = el.enabled === false; commit(); afterMutate(); }
            else if (ed === "duplink") linkDuplicate(el.id);
            else if (ed === "unlink") unlink(el.id);
        });

        // scene fields
        root.addEventListener("input", (ev) => {
            if (ev.target.dataset.f === "hld") { state.high_level_description = ev.target.value; push(); return; }
            if (ev.target.dataset.f === "bg") { state.background = ev.target.value; push(); return; }
            if (ev.target.dataset.style) { state.style[ev.target.dataset.style] = ev.target.value; push(); return; }
        });
        root.addEventListener("change", (ev) => { if (ev.target.dataset.f || ev.target.dataset.style) commit(); });

        // ------------------------------------------------------------------ stage interaction
        let drag = null, lastPoint = { x: 500, y: 500 };
        const pt = (ev) => {
            const r = stage.getBoundingClientRect();
            return { x: clamp(((ev.clientX - r.left) / r.width) * 1000), y: clamp(((ev.clientY - r.top) / r.height) * 1000) };
        };
        const hint = (t) => { q("#hintbar", root).textContent = t || DEFAULT_HINT; };
        const DEFAULT_HINT = q("#hintbar", root).textContent;
        const bboxLabel = (b) => `bbox [y${b[0]} x${b[1]} y${b[2]} x${b[3]}] · ${b[3] - b[1]}×${b[2] - b[0]}`;

        stage.addEventListener("pointerdown", (ev) => {
            if (ev.button !== 0) return;
            if (ev.target.closest(".id4s-inline") || ev.target.closest(".id4s-menu")) return;
            ev.preventDefault();
            try { stage.setPointerCapture(ev.pointerId); } catch { }
            stage.focus({ preventScroll: true });
            const p = pt(ev);
            const handle = ev.target.dataset.handle;
            const isGroupHandle = ev.target.dataset.group;
            const boxEl = ev.target.closest("[data-box]");
            const boxId = boxEl ? boxEl.dataset.box : null;
            const mod = ev.ctrlKey || ev.metaKey;

            if (handle && isGroupHandle) {
                const gb = groupBox();
                if (gb) { drag = { kind: "resize-group", handle, start: p, groupStart: gb.slice(), groupTargets: selectedBoxes().map(e => ({ el: e, start: e.bbox.slice() })) }; return; }
            }
            if (handle && primaryId() && selectedIds.length === 1) {
                const el = getElement(primaryId());
                if (el && el.bbox) { drag = { kind: "resize", el, handle, start: p }; return; }
            }
            if (mod) {
                // Ctrl/Meta always draws (can start over an existing box)
                drag = { kind: "draw", start: p, ctrlClick: true, clickedId: boxId, moved: false };
                return;
            }
            if (boxId) {
                if (ev.shiftKey) { select(boxId, true); return; }
                const wasPrimary = selectedIds.length === 1 && primaryId() === boxId;
                const wasClickedPrimary = primaryId() === boxId;
                if (!isSelected(boxId)) select(boxId);
                const targets = selectedIds.map(getElement).filter(e => e && e.bbox).map(e => ({ el: e, start: e.bbox.slice() }));
                drag = { kind: "move", start: p, moved: false, targets, clickedId: boxId, wasPrimary, wasClickedPrimary };
                return;
            }
            if (ev.shiftKey) { drag = { kind: "marquee", start: p, moved: false, additive: true }; return; }
            // plain drag on empty canvas draws (like Comfy studio)
            drag = { kind: "draw", start: p, ctrlClick: false, clickedId: null, moved: false };
        });

        stage.addEventListener("pointermove", (ev) => {
            const p = pt(ev); lastPoint = p;
            if (!drag) return;
            ev.preventDefault();
            if (Math.abs(p.x - drag.start.x) > 2 || Math.abs(p.y - drag.start.y) > 2) drag.moved = true;
            if (drag.kind === "draw") {
                drag.cur = [Math.min(drag.start.y, p.y), Math.min(drag.start.x, p.x), Math.max(drag.start.y, p.y), Math.max(drag.start.x, p.x)];
                setDraft(drag.cur); hint(bboxLabel(drag.cur));
            } else if (drag.kind === "marquee") {
                setDraft(drag.cur = [Math.min(drag.start.y, p.y), Math.min(drag.start.x, p.x), Math.max(drag.start.y, p.y), Math.max(drag.start.x, p.x)], true);
            } else if (drag.kind === "resize-group") {
                const [oy0, ox0, oy1, ox1] = drag.groupStart;
                let gy0 = oy0, gx0 = ox0, gy1 = oy1, gx1 = ox1;
                if (drag.handle.includes("n")) gy0 = Math.min(p.y, oy1 - MIN_SPAN);
                if (drag.handle.includes("s")) gy1 = Math.max(p.y, oy0 + MIN_SPAN);
                if (drag.handle.includes("w")) gx0 = Math.min(p.x, ox1 - MIN_SPAN);
                if (drag.handle.includes("e")) gx1 = Math.max(p.x, ox0 + MIN_SPAN);
                const sx = ox1 - ox0 ? (gx1 - gx0) / (ox1 - ox0) : 1, sy = oy1 - oy0 ? (gy1 - gy0) / (oy1 - oy0) : 1;
                for (const t of drag.groupTargets) {
                    const [y0, x0, y1, x1] = t.start;
                    t.el.bbox = [clamp(gy0 + (y0 - oy0) * sy), clamp(gx0 + (x0 - ox0) * sx), clamp(gy0 + (y1 - oy0) * sy), clamp(gx0 + (x1 - ox0) * sx)];
                }
                renderBoxes();
            } else if (drag.kind === "move") {
                let dx = p.x - drag.start.x, dy = p.y - drag.start.y;
                let loX = -1e9, hiX = 1e9, loY = -1e9, hiY = 1e9;
                for (const t of drag.targets) {
                    const [y0, x0, y1, x1] = t.start;
                    loX = Math.max(loX, -x0); hiX = Math.min(hiX, 1000 - x1);
                    loY = Math.max(loY, -y0); hiY = Math.min(hiY, 1000 - y1);
                }
                dx = Math.max(loX, Math.min(hiX, dx)); dy = Math.max(loY, Math.min(hiY, dy));
                for (const t of drag.targets) {
                    const [y0, x0, y1, x1] = t.start;
                    t.el.bbox = [clamp(y0 + dy), clamp(x0 + dx), clamp(y1 + dy), clamp(x1 + dx)];
                }
                if (drag.targets.length === 1) hint(bboxLabel(drag.targets[0].el.bbox));
                renderBoxes();
            } else if (drag.kind === "resize") {
                let [y0, x0, y1, x1] = drag.el.bbox;
                if (drag.handle.includes("n")) y0 = Math.min(p.y, y1 - 2);
                if (drag.handle.includes("s")) y1 = Math.max(p.y, y0 + 2);
                if (drag.handle.includes("w")) x0 = Math.min(p.x, x1 - 2);
                if (drag.handle.includes("e")) x1 = Math.max(p.x, x0 + 2);
                drag.el.bbox = [clamp(y0), clamp(x0), clamp(y1), clamp(x1)];
                hint(bboxLabel(drag.el.bbox));
                renderBoxes();
            }
        });

        stage.addEventListener("pointerup", (ev) => {
            if (!drag) return;
            try { stage.releasePointerCapture(ev.pointerId); } catch { }
            if (drag.kind === "draw") {
                const b = drag.cur; clearDraft();
                if (b && b[2] - b[0] >= 8 && b[3] - b[1] >= 8) addElement(newType, b);
                else if (!drag.moved && drag.ctrlClick && drag.clickedId) select(drag.clickedId, true);
                else if (!drag.moved && !drag.ctrlClick) select(null);
            } else if (drag.kind === "marquee") {
                const m = drag.cur; clearDraft();
                if (drag.moved && m) {
                    const ids = state.elements.filter(el => el.bbox && el.enabled !== false &&
                        el.bbox[1] < m[3] && el.bbox[3] > m[1] && el.bbox[0] < m[2] && el.bbox[2] > m[0]).map(el => el.id);
                    setSelection(ids, drag.additive);
                } else if (!drag.moved && !drag.additive) select(null);
            } else if (drag.kind === "move" && !drag.moved && drag.clickedId) {
                const el = getElement(drag.clickedId);
                if (el && el.groupId) {
                    if (drag.wasClickedPrimary) select(null); else select(drag.clickedId);
                } else if (selectedIds.length > 1) select(drag.clickedId);
                else if (drag.wasPrimary) {
                    const ids = boxesUnder(drag.start.x, drag.start.y);
                    if (ids.length > 1) {
                        const ci = ids.indexOf(drag.clickedId);
                        select(ids[ci === -1 ? 0 : (ci + 1) % ids.length]);
                    }
                }
            }
            drag = null; hint(null); commit();
        });

        function setDraft(b, marquee) {
            clearDraft();
            const d = document.createElement("div");
            d.className = "id4s-draft" + (marquee ? " marquee" : "");
            d.style.cssText = `left:${b[1] / 10}%;top:${b[0] / 10}%;width:${(b[3] - b[1]) / 10}%;height:${(b[2] - b[0]) / 10}%`;
            boxesHost.appendChild(d);
        }
        function clearDraft() { boxesHost.querySelectorAll(".id4s-draft").forEach(n => n.remove()); }

        // double-click => inline prompt input (the requested Comfy-like behavior)
        stage.addEventListener("dblclick", (ev) => {
            // the first click re-renders boxes, so dblclick's target is the
            // stage (common ancestor) - hit-test by coordinates instead
            let boxEl = ev.target.closest("[data-box]");
            if (!boxEl) {
                const hit = document.elementFromPoint(ev.clientX, ev.clientY);
                boxEl = hit && hit.closest ? hit.closest("[data-box]") : null;
            }
            if (!boxEl) return;
            const el = getElement(boxEl.dataset.box); if (!el) return;
            select(el.id);
            openInlineEditor(el);
        });
        let inline = null;
        function openInlineEditor(el) {
            closeInline();
            const [y0, x0, y1, x1] = el.bbox;
            const ta = document.createElement("textarea");
            ta.className = "id4s-inline";
            ta.value = el.type === "text" ? (el.text || "") : (el.desc || "");
            ta.placeholder = el.type === "text" ? "text to render…" : "prompt for this region…";
            ta.style.cssText = `left:${x0 / 10}%;top:${y0 / 10}%;width:${(x1 - x0) / 10}%;height:${(y1 - y0) / 10}%;border-color:${el.boxColor}`;
            // inline styles beat any global/cached Gradio CSS - the editor
            // must stay matte dark glass with readable light text
            ta.style.background = "rgba(16,18,22,.55)";
            ta.style.backdropFilter = "blur(6px)";
            ta.style.color = "#fff";
            stage.appendChild(ta); ta.focus(); ta.select();
            inline = { el, ta };
            ta.addEventListener("input", () => setShared(el.id, el.type === "text" ? "text" : "desc", ta.value));
            ta.addEventListener("keydown", (ev2) => {
                ev2.stopPropagation();
                if (ev2.key === "Enter" && !ev2.shiftKey) { ev2.preventDefault(); closeInline(); commit(); }
                if (ev2.key === "Escape") { closeInline(); }
            });
            ta.addEventListener("blur", () => { closeInline(); commit(); });
        }
        function closeInline() { if (inline) { inline.ta.remove(); inline = null; } }

        // right-click menu
        let ctxMenu = null;
        stage.addEventListener("contextmenu", (ev) => {
            ev.preventDefault();
            const boxEl = ev.target.closest("[data-box]");
            closeCtx();
            if (!boxEl) return;
            const id = boxEl.dataset.box;
            if (!isSelected(id)) select(id);
            const el = getElement(id); if (!el) return;
            const items = [
                [el.type === "text" ? "make object" : "make text", () => { setShared(el.id, "type", el.type === "text" ? "obj" : "text"); renderEditor(); }],
                [el.enabled === false ? "unmute" : "mute", () => { el.enabled = el.enabled === false; commit(); afterMutate(); }],
                ["linked copy", () => linkDuplicate(el.id)],
                [el.groupId ? "ungroup" : "group selection", () => el.groupId ? ungroup(el.id) : groupSel()],
                ["duplicate", () => { pasteAtAt((el.bbox[1] + el.bbox[3]) / 2 + 50, (el.bbox[0] + el.bbox[2]) / 2 + 50); }],
                ["delete", () => removeElements([...selectedIds])],
            ];
            ctxMenu = document.createElement("div");
            ctxMenu.className = "id4s-menu ctx";
            for (const [label, fn] of items) {
                const b = document.createElement("button");
                b.textContent = label;
                b.addEventListener("click", () => { closeCtx(); fn(); });
                ctxMenu.appendChild(b);
            }
            const r = stage.getBoundingClientRect();
            showMenu(ctxMenu, ev.clientX, ev.clientY);
        });
        function closeCtx() { if (ctxMenu) { ctxMenu.remove(); ctxMenu = null; } }

        // menus live outside the stage (fixed, viewport coords) so they are
        // never clipped by its overflow:hidden; clamped into the viewport
        function showMenu(menu, x, y) {
            root.appendChild(menu);
            menu.style.left = x + "px"; menu.style.top = y + "px";
            const mw = menu.offsetWidth, mh = menu.offsetHeight;
            const lx = Math.max(8, Math.min(x, innerWidth - mw - 8));
            const ly = Math.max(8, Math.min(y, innerHeight - mh - 8));
            menu.style.left = lx + "px"; menu.style.top = ly + "px";
        }
        document.addEventListener("pointerdown", (ev) => {
            if (ctxMenu && !ev.target.closest(".id4s-menu")) closeCtx();
            // the aspect/size menu also closes on any click outside it
            // (clicks inside the menu or on its own button keep it open)
            const rm = q("#res-menu", root);
            if (rm && !rm.hidden && !ev.target.closest("#res-menu") && !ev.target.closest("#res-btn")) rm.hidden = true;
        });
        function pasteAtAt(x, y) { copySel(); pasteAt(x, y); }

        // keyboard (only while canvas focused)
        stage.addEventListener("keydown", (ev) => {
            if (inline) return;
            const mod = ev.ctrlKey || ev.metaKey;
            if (mod && "zZ".includes(ev.key)) { ev.preventDefault(); ev.shiftKey ? redo() : undo(); return; }
            if (mod && "yY".includes(ev.key)) { ev.preventDefault(); redo(); return; }
            if (mod && "cC".includes(ev.key)) { ev.preventDefault(); copySel(); return; }
            if (mod && "vV".includes(ev.key)) { ev.preventDefault(); pasteAt(lastPoint.x, lastPoint.y); return; }
            if (mod && "gG".includes(ev.key)) {
                ev.preventDefault();
                if (ev.shiftKey) { const id = primaryId(); if (id) ungroup(id); }
                else groupSel();
                return;
            }
            if ((ev.key === "Delete" || ev.key === "Backspace") && selectedIds.length) {
                ev.preventDefault(); removeElements([...selectedIds]); return;
            }
            if (ev.key === "Escape") { select(null); return; }
        });

        // ------------------------------------------------------------------ toolbar
        q("#tb", root).addEventListener("click", (ev) => {
            const btn = ev.target.closest("button");
            const act = btn && btn.dataset.act;
            if (!act) return;
            if (act === "undo") undo();
            else if (act === "redo") redo();
            else if (act === "newtype") {
                newType = newType === "obj" ? "text" : "obj";
                btn.innerHTML = newType === "text" ? "T text" : "▢ obj";
            }
            else if (act === "ref") q("#ref-file", root).click();
            else if (act === "labels") {
                showLabels = !showLabels;
                ev.target.classList.toggle("on", showLabels);
                renderBoxes();
            }
            else if (act === "fliph") flipAll("h");
            else if (act === "flipv") flipAll("v");
            else if (act === "res") {
                const m = q("#res-menu", root);
                m.hidden = !m.hidden;
                if (!m.hidden) {
                    // fixed-position menu anchored under the button, clamped to viewport
                    const br = btn.getBoundingClientRect();
                    showMenu(m, br.left, br.bottom + 4);
                }
            }
            else if (act === "wh-swap") { setRes(resH, resW); }
            else if (act === "preset-official" || act === "preset-comfy") {
                const official = act === "preset-official";
                // clicking the already-active preset button turns it OFF:
                // the time-shift then returns to the engine defaults; the
                // normal Forge settings keep whatever the user has there
                if (state.preset === (official ? "official" : "comfy")) {
                    state.preset = null;
                    for (const b of q("#tb", root).querySelectorAll("[data-act^='preset']"))
                        b.classList.remove("on");
                    hint("Preset cleared — the sampler time-shift returns to model defaults on the next generation. Forge settings stay as they are.");
                    push();
                    return;
                }
                // A studio preset is an INPUT into the normal Forge controls:
                // we write sampler/scheduler/steps/CFG into the real UI of the
                // tab. Generation follows whatever ends up there afterwards,
                // and the user can adjust everything freely - the studio never
                // overrides the Forge settings at run time.
                const missing = [];
                if (!setTabField(tab, "steps", 20)) missing.push("steps");
                if (!setTabField(tab, "cfg_scale", official ? 7 : 3)) missing.push("cfg");
                // dropdowns resolve asynchronously (filter list must render):
                // chain sampler -> scheduler, then report honestly
                setTabDropdown(tab, "sampling", official ? "Euler" : "Res Multistep", (ok1) => {
                    if (!ok1) missing.push("sampling");
                    setTabDropdown(tab, "scheduler", "Simple", (ok2) => {
                        if (!ok2) missing.push("scheduler");
                        hint(missing.length
                            ? `Preset set, but NOT found in UI: ${missing.join(", ")} — set those manually.`
                            : (official
                                ? "Preset OFFICIAL set in the Forge controls: Euler / Simple / 20 / CFG 7 (the engine polishes the last steps to 3), time-shift mu=0.0. Adjust anything freely."
                                : "Preset COMFY STUDIO set in the Forge controls: Res Multistep / Simple / 20 / CFG 3, time-shift mu=0.5. Adjust anything freely."));
                    });
                });
                // the state marker now carries only what the UI cannot show:
                // the logit-normal time-shift (official mu=0.0 / comfy mu=0.5)
                state.preset = official ? "official" : "comfy";
                for (const b of q("#tb", root).querySelectorAll("[data-act^='preset']"))
                    b.classList.toggle("on", b.dataset.act === act);
                push();
            }
        });
        root.addEventListener("click", (ev) => {
            const act = ev.target.dataset.act;
            if (act === "add-obj") addElement("obj");
            else if (act === "add-text") addElement("text");
        });
        q("#ref-file", root).addEventListener("change", (ev) => {
            const f = ev.target.files && ev.target.files[0]; if (!f) return;
            const r = new FileReader();
            r.onload = () => { backdrop.src = r.result; backdrop.hidden = false; };
            r.readAsDataURL(f);
        });
        q("#ref-op", root).addEventListener("input", (ev) => { backdrop.style.opacity = ev.target.value; });
        stage.addEventListener("dragover", (ev) => ev.preventDefault());
        stage.addEventListener("drop", (ev) => {
            ev.preventDefault();
            const f = ev.dataTransfer && ev.dataTransfer.files[0];
            if (f && f.type.startsWith("image/")) {
                const r = new FileReader();
                r.onload = () => { backdrop.src = r.result; backdrop.hidden = false; };
                r.readAsDataURL(f);
            }
        });
        q("#el-filter", root).addEventListener("change", (ev) => { filterMode = ev.target.value; renderList(); });

        // resolution menu
        const aspHost = q("#asp-chips", root);
        for (const [label, ratio] of ASPECTS) {
            const b = document.createElement("button");
            b.className = "id4s-achip"; b.dataset.ratio = ratio; b.dataset.label = label;
            const wpx = Math.min(34, 24 * Math.max(1, ratio)), hpx = Math.min(34, 24 / Math.max(1, 1 / ratio));
            b.innerHTML = `<span style="display:inline-block;width:${Math.round(wpx)}px;height:${Math.round(hpx)}px;border:1px solid currentColor;border-radius:2px;margin-right:4px;vertical-align:middle"></span>${label}`;
            b.addEventListener("click", () => {
                const mp = Math.max(0.5, (resW * resH) / 1e6);
                let w = Math.round(Math.sqrt(mp * 1e6 * ratio) / 8) * 8;
                let h = Math.round(w / ratio / 8) * 8;
                w = Math.max(64, Math.min(2048, w)); h = Math.max(64, Math.min(2048, h));
                setRes(w, h);
            });
            aspHost.appendChild(b);
        }
        function setRes(w, h) {
            // explicit values are authoritative - never re-read inputs here
            resW = w; resH = h;
            setGenSize(tab, resW, resH);   // write into the tab sliders too
            syncRes();
        }
        function applyRes() {
            // user edited our W/H number fields
            const w = Math.max(16, Math.min(2048, parseInt(q("#wh-w", root).value) || 1024));
            const h = Math.max(16, Math.min(2048, parseInt(q("#wh-h", root).value) || 1024));
            setRes(w, h);
        }
        q("#wh-w", root).addEventListener("change", applyRes);
        q("#wh-h", root).addEventListener("change", applyRes);

        // follow genuine user edits of the tab's own width/height sliders
        for (const id of (tab === "i2i" ? ["#img2img_width", "#img2img_height"] : ["#txt2img_width", "#txt2img_height"])) {
            const el = q(id + " input") || q(id);
            if (!el) continue;
            el.addEventListener("change", () => {
                const g = genSize(tab);
                if (g.w !== resW || g.h !== resH) { resW = g.w; resH = g.h; syncRes(); }
            });
        }

        // ------------------------------------------------------------------ restore inputs & boot
        for (const e of state.elements) {
            // sanity
            if (e.bbox) e.bbox = e.bbox.map(clamp);
            e.enabled = e.enabled !== false;
            e.color_palette = (Array.isArray(e.color_palette) ? e.color_palette : []).slice(0, 5);
        }
        if (state.overlay && state.overlay.showLabels === false) { showLabels = false; q("#labels-btn", root).classList.remove("on"); }
        if (state.preset) {
            const act = state.preset === "official" ? "preset-official" : "preset-comfy";
            const b = q(`#tb [data-act="${act}"]`, root);
            if (b) b.classList.add("on");
        }
        backdrop.style.opacity = 0.6;
        // adopt the tab's current size as the initial aspect, unless the studio
        // already remembers one from a previous session
        const g0 = genSize(tab);
        if (state.res && state.res.w && state.res.h) { resW = state.res.w; resH = state.res.h; }
        else { resW = g0.w; resH = g0.h; }
        syncRes();
        new ResizeObserver(() => renderBoxes()).observe(stage);
        renderBoxes(); renderList(); renderEditor(); updateCounts(); doPush();
    }

    function boot() { bootStudio("t2i"); bootStudio("i2i"); }

    // This script loads in <head>, before <body> exists (Forge injects
    // extension JS there). Everything DOM-related must wait for body.
    let tries = 0;
    const iv = setInterval(() => { boot(); if (++tries > 40) clearInterval(iv); }, 1000);
    const whenBody = (fn) => {
        if (document.body) return fn();
        document.addEventListener("DOMContentLoaded", fn, { once: true });
    };
    whenBody(() => {
        boot();
        // Gradio re-renders HTML blocks (accordion toggle, tab switch) and wipes
        // injected DOM; the host element is recreated fresh each time. Watch for
        // hosts without our app and mount again - state restores from localStorage.
        new MutationObserver(() => boot()).observe(document.body, { childList: true, subtree: true });
    });
})();

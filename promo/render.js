// Renders a promo page (PAGE=ad.html by default, DUR = its length in seconds) frame by frame with headless Chromium.
//   node render.js stills <fmt h|v> <outDir> <t1,t2,...>
//   node render.js frames <fmt h|v> <outDir> <fps> [workers] [samples]
//     writes <samples> sub-frames per output frame (180° shutter), blended later by ffmpeg tmix
const http = require('http');
const fs = require('fs');
const path = require('path');
const { chromium } = require('playwright');

const ROOT = __dirname;
const TYPES = { '.html': 'text/html; charset=utf-8', '.js': 'text/javascript', '.css': 'text/css', '.png': 'image/png', '.jpg': 'image/jpeg', '.woff2': 'font/woff2', '.m4a': 'audio/mp4', '.wav': 'audio/wav' };

function serve() {
  return new Promise(res => {
    const srv = http.createServer((req, rsp) => {
      const p = path.join(ROOT, decodeURIComponent(req.url.split('?')[0]));
      if (!p.startsWith(ROOT) || !fs.existsSync(p) || fs.statSync(p).isDirectory()) { rsp.writeHead(404); return rsp.end(); }
      rsp.writeHead(200, { 'Content-Type': TYPES[path.extname(p)] || 'application/octet-stream' });
      fs.createReadStream(p).pipe(rsp);
    });
    srv.listen(0, '127.0.0.1', () => res(srv));
  });
}

async function openPage(browser, port, fmt) {
  const [w, h] = fmt === 'v' ? [1080, 1920] : [1920, 1080];
  const page = await browser.newPage({ viewport: { width: w, height: h }, deviceScaleFactor: 1 });
  page.on('pageerror', e => console.error('PAGE ERROR', e.message));
  page.on('console', m => { if (m.type() === 'error') console.error('CONSOLE', m.text()); });
  await page.goto(`http://127.0.0.1:${port}/${process.env.PAGE || 'ad.html'}?render=1&fmt=${fmt}`, { waitUntil: 'load' });
  await page.evaluate(() => window.__ready);
  return page;
}

async function grab(page, t, file, gi) {
  await page.evaluate(([t, gi]) => window.__seek(t, gi), [t, gi]);
  const jpg = file.endsWith('.jpg');
  await page.screenshot({ path: file, type: jpg ? 'jpeg' : 'png', quality: jpg ? 95 : undefined, animations: 'allow', caret: 'initial' });
}

(async () => {
  const [mode, fmt, outDir, arg, workersArg, fromArg, toArg] = process.argv.slice(2);
  fs.mkdirSync(outDir, { recursive: true });
  const srv = await serve();
  const port = srv.address().port;
  const browser = await chromium.launch({ args: ['--font-render-hinting=none', '--disable-lcd-text', '--force-color-profile=srgb'] });
  try {
    if (mode === 'stills') {
      const page = await openPage(browser, port, fmt);
      for (const ts of arg.split(',')) {
        const t = parseFloat(ts);
        await grab(page, t, path.join(outDir, `t${t.toFixed(2).padStart(5, '0')}.png`));
      }
    } else {
      const fps = parseInt(arg, 10);
      const S = parseInt(fromArg || '6', 10);
      const dur = parseFloat(process.env.DUR || '30');
      const total = Math.round(dur * fps) * S;
      const workers = parseInt(workersArg || '4', 10);
      const expo = 0.5 / fps;               // 180° shutter
      // optional output-frame range, e.g. FROM=420 TO=490 to re-render one passage
      const lo = (+process.env.FROM || 0) * S, hi = process.env.TO ? +process.env.TO * S : total;
      let next = lo, done = 0;
      const t0 = Date.now();
      await Promise.all(Array.from({ length: workers }, async () => {
        const page = await openPage(browser, port, fmt);
        while (next < hi) {
          const g = next++;
          const i = Math.floor(g / S), j = g % S;
          const t = Math.min(dur - 1e-3, Math.max(0, i / fps + ((j + 0.5) / S - 0.5) * expo));
          const file = path.join(outDir, `s${String(g).padStart(6, '0')}.jpg`);
          if (!fs.existsSync(file)) await grab(page, t, file, i);
          if (++done % 300 === 0) console.log(`${done}/${total} sub-frames · ${((Date.now() - t0) / done).toFixed(0)} ms each`);
        }
        await page.close();
      }));
      console.log(`done ${done} frames in ${((Date.now() - t0) / 1000).toFixed(1)}s`);
    }
  } finally {
    await browser.close();
    srv.close();
  }
})();

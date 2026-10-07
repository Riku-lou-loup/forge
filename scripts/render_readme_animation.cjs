const { chromium } = require(process.env.FORGE_PLAYWRIGHT_PATH || 'playwright');
const fs = require('node:fs');
const path = require('node:path');
const { pathToFileURL } = require('node:url');
const root = path.resolve(__dirname, '..');
const work = path.join(root, '.cache', 'readme-motion');
const source = path.join(root, 'docs', 'assets', 'workflow.html');
(async () => {
  const browser = await chromium.launch({ ...(process.env.FORGE_BROWSER_BINARY ? { executablePath: process.env.FORGE_BROWSER_BINARY } : {}), headless: true, args: ['--disable-gpu', '--renderer-process-limit=1'] });
  const page = await browser.newPage({ viewport: { width: 1200, height: 460 } });
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  for (const theme of ['light', 'dark']) {
    const dir = path.join(work, 'frames', theme);
    fs.mkdirSync(dir, { recursive: true });
    await page.goto(pathToFileURL(source).href + '?capture&theme=' + theme);
    for (let frame = 0; frame < 360; frame++) {
      const png = await page.evaluate(time => {
        window.renderForge(time);
        return document.querySelector('canvas').toDataURL('image/png').split(',')[1];
      }, frame / 25);
      fs.writeFileSync(path.join(dir, String(frame).padStart(3, '0') + '.png'), Buffer.from(png, 'base64'));
    }
    console.log(`Rendered ${theme}: 360 frames at 25 fps`);
  }
  const tokens = await page.evaluate(() => window.motionTokens);
  fs.writeFileSync(path.join(work, 'motion-tokens.json'), JSON.stringify(tokens, null, 2) + '\n');
  const client = await page.context().newCDPSession(page);
  await client.send('Emulation.setCPUThrottlingRate', { rate: 4 });
  const perf = await page.evaluate(() => {
    const durations = [];
    for (let i = 0; i < 100; i++) { const start = performance.now(); window.renderForge(i * .144); durations.push(performance.now() - start); }
    return { mean_ms: durations.reduce((a,b) => a+b, 0) / durations.length, p95_ms: durations.sort((a,b) => a-b)[94] };
  });
  await client.send('Emulation.setCPUThrottlingRate', { rate: 1 });
  await page.emulateMedia({ reducedMotion: 'reduce' });
  await page.goto(pathToFileURL(source).href + '?theme=dark');
  const before = await page.locator('canvas').evaluate(canvas => canvas.toDataURL());
  await page.waitForTimeout(250);
  const after = await page.locator('canvas').evaluate(canvas => canvas.toDataURL());
  if (before !== after) throw new Error('Reduced-motion frame changed');
  if (await page.getByRole('button').innerText() !== 'Play') throw new Error('Reduced motion did not pause');
  await page.emulateMedia({ reducedMotion: 'no-preference' });
  await page.goto(pathToFileURL(source).href + '?theme=dark');
  await page.getByRole('button').click();
  if (await page.getByRole('button').innerText() !== 'Play') throw new Error('Pause failed');
  if (errors.length) throw new Error(errors.join('\n'));
  fs.writeFileSync(path.join(work, 'browser-validation.json'), JSON.stringify({ perf_at_4x_cpu: perf, reducedMotion: 'static', pause: 'passed', errors }, null, 2));
  console.log(JSON.stringify({ perf_at_4x_cpu: perf, reducedMotion: 'static', pause: 'passed' }));
  await browser.close();
})().catch(error => { console.error(error); process.exit(1); });

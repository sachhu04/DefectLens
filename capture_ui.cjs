const puppeteer = require('puppeteer');
const path = require('path');

(async () => {
    try {
        const browser = await puppeteer.launch();
        const page = await browser.newPage();
        
        await page.setViewport({ width: 1440, height: 900 });
        await page.goto('http://localhost:5173', { waitUntil: 'networkidle0' });
        
        // Remove hidden class so Puppeteer can interact flawlessly
        await page.evaluate(() => {
            document.querySelector('input[type="file"]').classList.remove('hidden');
        });
        
        const filePath = path.resolve('data/mvtec/bottle/test/broken_large/000.png');
        
        // Find input and upload
        const fileInput = await page.$('input[type="file"]');
        await fileInput.uploadFile(filePath);
        
        // Wait a second for React state
        await new Promise(r => setTimeout(r, 1000));
        
        // Hide the input again so it looks good in the screenshot
        await page.evaluate(() => {
            document.querySelector('input[type="file"]').classList.add('hidden');
        });
        
        // Click inspect button
        await page.evaluate(() => {
            const btns = Array.from(document.querySelectorAll('button'));
            const inspectBtn = btns.find(b => b.textContent.includes('Run Inspection'));
            if (inspectBtn && !inspectBtn.disabled) {
                inspectBtn.click();
            } else {
                throw new Error("Button disabled or not found");
            }
        });
        
        console.log('Waiting for inference...');
        await page.waitForSelector('img[src*="data:image/png;base64"]', { timeout: 30000 });
        // Wait for CSS fade-in / slide-up transitions to finish completely
        await new Promise(r => setTimeout(r, 1500));
        
        await page.screenshot({ path: 'real_screenshot_ui_results.png' });
        console.log('Successfully captured UI screenshot!');
        
        await browser.close();
    } catch (err) {
        console.error('Error:', err);
        process.exit(1);
    }
})();

const puppeteer = require('puppeteer');
const path = require('path');

(async () => {
    try {
        console.log('Launching Puppeteer browser...');
        const browser = await puppeteer.launch({
            headless: 'new',
            args: ['--no-sandbox', '--disable-setuid-sandbox']
        });

        const page = await browser.newPage();
        const htmlPath = path.resolve(__dirname, 'DefectLens_Complete_Technical_Guide_and_Viva_Prep.html');
        console.log(`Loading HTML from: ${htmlPath}`);

        await page.goto(`file://${htmlPath}`, { waitUntil: ['load', 'networkidle0'], timeout: 60000 });
        await page.emulateMediaType('screen');

        // Allow KaTeX fonts and formulas to finish rendering
        await page.evaluate(() => new Promise(resolve => setTimeout(resolve, 1500)));

        const pdfPath = path.resolve(__dirname, 'DefectLens_Complete_Technical_Guide_and_Viva_Prep.pdf');
        console.log(`Rendering PDF to: ${pdfPath}...`);

        await page.pdf({
            path: pdfPath,
            format: 'A4',
            printBackground: true,
            margin: { top: '0.45in', bottom: '0.45in', left: '0.45in', right: '0.45in' },
            displayHeaderFooter: true,
            headerTemplate: `
                <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; font-size: 7.5pt; width: 100%; display: flex; justify-content: space-between; padding: 0 0.45in; color: #64748b; border-bottom: 1px solid #e2e8f0; padding-bottom: 3px;">
                    <span style="font-weight: 600; color: #0f172a;">DefectLens &bull; ViT Anomaly Detection with MCTF</span>
                    <span>CSE411 Computer Vision &bull; Comprehensive Technical Guide & Viva Prep</span>
                </div>
            `,
            footerTemplate: `
                <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; font-size: 7.5pt; width: 100%; display: flex; justify-content: space-between; padding: 0 0.45in; color: #64748b; border-top: 1px solid #e2e8f0; padding-top: 3px;">
                    <span>Team 7 &bull; IIT / CSE &bull; github.com/sachhu04/DefectLens</span>
                    <span style="font-weight: 600;">Page <span class="pageNumber"></span> of <span class="totalPages"></span></span>
                </div>
            `
        });

        console.log(`PDF successfully generated: ${pdfPath}`);
        await browser.close();
    } catch (err) {
        console.error('Error generating PDF:', err);
        process.exit(1);
    }
})();

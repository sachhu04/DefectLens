const puppeteer = require('puppeteer');
const path = require('path');

(async () => {
    try {
        console.log('Launching browser for PDF generation...');
        const browser = await puppeteer.launch({
            headless: 'new',
            args: ['--no-sandbox', '--disable-setuid-sandbox']
        });

        // 1. Primary Submission PDF: Space Grotesk 3-Page IEEE Format
        {
            const page = await browser.newPage();
            const htmlPath = path.resolve(__dirname, 'literature_survey_space_grotesk.html');
            console.log(`Loading Space Grotesk HTML: ${htmlPath}`);
            await page.goto(`file://${htmlPath}`, { waitUntil: ['load', 'networkidle0'], timeout: 60000 });
            await page.emulateMediaType('screen');
            const pdfPath = path.resolve(__dirname, 'LAB SUBMISSION CV LITERATURE SURVEY AND RESEARCH GAP TEAM 7.pdf');
            await page.pdf({
                path: pdfPath,
                format: 'A4',
                printBackground: true,
                margin: { top: '0.35in', bottom: '0.35in', left: '0.45in', right: '0.45in' }
            });
            console.log(`Saved: ${pdfPath}`);
            await page.close();
        }

        // 2. Times New Roman IEEE Format
        {
            const page = await browser.newPage();
            const htmlPath = path.resolve(__dirname, 'literature_survey_paper.html');
            await page.goto(`file://${htmlPath}`, { waitUntil: ['load', 'networkidle0'], timeout: 60000 });
            await page.emulateMediaType('screen');
            const pdfPath = path.resolve(__dirname, 'Literature_Survey_Research_Paper_Format.pdf');
            await page.pdf({
                path: pdfPath,
                format: 'A4',
                printBackground: true,
                margin: { top: '0.35in', bottom: '0.35in', left: '0.45in', right: '0.45in' }
            });
            console.log(`Saved: ${pdfPath}`);
            await page.close();
        }

        // 3. Extended Web-Style Format
        {
            const page = await browser.newPage();
            const htmlPath = path.resolve(__dirname, 'literature_survey.html');
            await page.goto(`file://${htmlPath}`, { waitUntil: ['load', 'networkidle0'], timeout: 60000 });
            await page.emulateMediaType('screen');
            const pdfPath = path.resolve(__dirname, 'Literature_Survey_DefectLens.pdf');
            await page.pdf({
                path: pdfPath,
                format: 'A4',
                printBackground: true,
                margin: { top: '0.4in', bottom: '0.4in', left: '0.4in', right: '0.4in' },
                displayHeaderFooter: true,
                headerTemplate: `
                    <div style="font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; font-size: 8pt; width: 100%; display: flex; justify-content: space-between; padding: 0 0.4in; color: #64748b; border-bottom: 1px solid #e2e8f0; padding-bottom: 4px;">
                        <span>DefectLens (Team 7) &bull; Literature Survey & Research Gap Analysis</span>
                        <span>Computer Vision Course Project</span>
                    </div>
                `,
                footerTemplate: `
                    <div style="font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; font-size: 8pt; width: 100%; display: flex; justify-content: space-between; padding: 0 0.4in; color: #64748b; border-top: 1px solid #e2e8f0; padding-top: 4px;">
                        <span>Rolls: 2023BCS0052, 2023BCS0064, 2023BCS0067, 2023BCS0202</span>
                        <span>Page <span class="pageNumber"></span> of <span class="totalPages"></span></span>
                    </div>
                `
            });
            console.log(`Saved: ${pdfPath}`);
            await page.close();
        }

        await browser.close();
        console.log('All Literature Survey PDFs generated successfully!');
    } catch (err) {
        console.error('Error generating PDF:', err);
        process.exit(1);
    }
})();

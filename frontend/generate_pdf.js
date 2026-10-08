const puppeteer = require('puppeteer');
const path = require('path');

(async () => {
    const browser = await puppeteer.launch();
    const page = await browser.newPage();
    const htmlPath = path.resolve('../report_draft.html');
    await page.goto(`file://${htmlPath}`, {waitUntil: 'networkidle0'});
    await page.pdf({
        path: '../Weekly_Project_Report.pdf',
        format: 'A4',
        printBackground: true,
        margin: { top: '20px', bottom: '20px', left: '20px', right: '20px' }
    });
    await browser.close();
    console.log('PDF generated at ../Weekly_Project_Report.pdf');
})();

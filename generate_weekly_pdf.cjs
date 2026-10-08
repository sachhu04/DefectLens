const puppeteer = require('puppeteer');
const path = require('path');

(async () => {
    const browser = await puppeteer.launch();
    const page = await browser.newPage();
    const htmlPath = path.resolve('weekly_report.html');
    await page.goto(`file://${htmlPath}`, {waitUntil: 'networkidle0'});
    await page.pdf({
        path: 'Weekly_Report_Research_Format.pdf',
        format: 'A4',
        printBackground: true,
        margin: { top: '0in', bottom: '0in', left: '0in', right: '0in' },
        preferCSSPageSize: true
    });
    await browser.close();
    console.log('PDF generated successfully.');
})();

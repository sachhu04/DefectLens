const puppeteer = require('puppeteer');
const path = require('path');

(async () => {
    const browser = await puppeteer.launch();
    const page = await browser.newPage();
    const htmlPath = path.resolve('Task4_Implementation_Report.html');
    await page.goto(`file://${htmlPath}`, {waitUntil: 'networkidle0'});
    await page.pdf({
        path: 'Task4_Implementation_Report.pdf',
        format: 'A4',
        printBackground: true,
        margin: { top: '0.5in', bottom: '0.5in', left: '0.5in', right: '0.5in' },
        preferCSSPageSize: true
    });
    await browser.close();
    console.log('PDF generated successfully: Task4_Implementation_Report.pdf');
})();

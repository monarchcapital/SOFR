<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PC Loading Heatmap (Outrights & Flies)</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        /* Custom styles for the chart */
        .chart-container {
            font-family: 'Inter', sans-serif;
            background-color: #ffffff;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
            max-width: 1200px;
            margin: 20px auto;
        }
        .group-separator {
            fill: none;
            stroke: #374151; /* Dark gray border */
            stroke-width: 2;
            shape-rendering: crispEdges;
        }
        .axis text {
            fill: #374151;
        }
        .tooltip {
            position: absolute;
            text-align: center;
            width: 120px;
            padding: 8px;
            background: #1f2937; /* Dark background */
            color: #f3f4f6; /* Light text */
            border: 0px;
            border-radius: 6px;
            pointer-events: none;
            font-size: 12px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
        }
    </style>
</head>
<body>

<div id="app" class="p-6 bg-gray-50 min-h-screen">
    <div class="chart-container">
        <h1 class="text-3xl font-bold mb-4 text-gray-800">SOFR PC Loading Heatmap: Outrights & Flies</h1>
        <div id="pc-summary" class="mb-6 grid grid-cols-1 md:grid-cols-4 gap-4">
            <!-- PC Summary will be injected here -->
        </div>
        <div id="heatmap" class="overflow-x-auto">
            <!-- D3 Heatmap will be injected here -->
        </div>
        <p class="mt-6 text-sm text-gray-600">
            **Note on Axes:** Outright contracts are labeled by their futures code (e.g., Z20, H21). Fly spreads are labeled by their component futures (e.g., Z20 / H21 / M21). The color intensity represents the magnitude of the loading (red for positive, blue for negative).
        </p>
    </div>
</div>

<script>
    // --- Data Injected from Python Analysis ---
    const heatmapData = [{"group":"Outrights","variable":"Z20","pc":"PC1","loading":-0.0381655181},{"group":"Outrights","variable":"Z20","pc":"PC2","loading":-0.0818276709},{"group":"Outrights","variable":"Z20","pc":"PC3","loading":-0.0401865961},{"group":"Outrights","variable":"H21","pc":"PC1","loading":-0.0813872583},{"group":"Outrights","variable":"H21","pc":"PC2","loading":-0.1293290656},{"group":"Outrights","variable":"H21","pc":"PC3","loading":-0.0308359281},{"group":"Outrights","variable":"M21","pc":"PC1","loading":-0.1384770144},{"group":"Outrights","variable":"M21","pc":"PC2","loading":-0.1652033588},{"group":"Outrights","variable":"M21","pc":"PC3","loading":-0.0210087796},{"group":"Outrights","variable":"U21","pc":"PC1","loading":-0.1868353982},{"group":"Outrights","variable":"U21","pc":"PC2","loading":-0.1601615822},{"group":"Outrights","variable":"U21","pc":"PC3","loading":-0.009095692},{"group":"Outrights","variable":"H22","pc":"PC1","loading":-0.2319245131},{"group":"Outrights","variable":"H22","pc":"PC2","loading":-0.1215112265},{"group":"Outrights","variable":"H22","pc":"PC3","loading":0.0039233633},{"group":"Outrights","variable":"M22","pc":"PC1","loading":-0.2705607062},{"group":"Outrights","variable":"M22","pc":"PC2","loading":-0.0673397395},{"group":"Outrights","variable":"M22","pc":"PC3","loading":0.0197771239},{"group":"Outrights","variable":"U22","pc":"PC1","loading":-0.3012543503},{"group":"Outrights","variable":"U22","pc":"PC2","loading":-0.0152914163},{"group":"Outrights","variable":"U22","pc":"PC3","loading":0.034732104},{"group":"Outrights","variable":"Z21","pc":"PC1","loading":-0.2974269191},{"group":"Outrights","variable":"Z21","pc":"PC2","loading":-0.0142345037},{"group":"Outrights","variable":"Z21","pc":"PC3","loading":0.0360655513},{"group":"Outrights","variable":"Z22","pc":"PC1","loading":-0.3061247953},{"group":"Outrights","variable":"Z22","pc":"PC2","loading":0.0084535805},{"group":"Outrights","variable":"Z22","pc":"PC3","loading":0.0385966952},{"group":"Outrights","variable":"H23","pc":"PC1","loading":-0.2443015488},{"group":"Outrights","variable":"H23","pc":"PC2","loading":0.0934185797},{"group":"Outrights","variable":"H23","pc":"PC3","loading":0.0637151593},{"group":"Outrights","variable":"M23","pc":"PC1","loading":-0.2307567781},{"group":"Outrights","variable":"M23","pc":"PC2","loading":0.1119770809},{"group":"Outrights","variable":"M23","pc":"PC3","loading":0.0682136009},{"group":"Outrights","variable":"U23","pc":"PC1","loading":-0.2173516541},{"group":"Outrights","variable":"U23","pc":"PC2","loading":0.1287606399},{"group":"Outrights","variable":"U23","pc":"PC3","loading":0.0718501178},{"group":"Outrights","variable":"Z23","pc":"PC1","loading":-0.2033008488},{"group":"Outrights","variable":"Z23","pc":"PC2","loading":0.1466033486},{"group":"Outrights","variable":"Z23","pc":"PC3","loading":0.0754877478},{"group":"Outrights","variable":"H24","pc":"PC1","loading":-0.1901358327},{"group":"Outrights","variable":"H24","pc":"PC2","loading":0.160759021},{"group":"Outrights","variable":"H24","pc":"PC3","loading":0.0783307521},{"group":"Outrights","variable":"M24","pc":"PC1","loading":-0.1774351052},{"group":"Outrights","variable":"M24","pc":"PC2","loading":0.1729221112},{"group":"Outrights","variable":"M24","pc":"PC3","loading":0.0807662888},{"group":"Outrights","variable":"U24","pc":"PC1","loading":-0.1652438865},{"group":"Outrights","variable":"U24","pc":"PC2","loading":0.1834165561},{"group":"Outrights","variable":"U24","pc":"PC3","loading":0.082729141},{"group":"Outrights","variable":"Z24","pc":"PC1","loading":-0.1534955949},{"group":"Outrights","variable":"Z24","pc":"PC2","loading":0.1925350751},{"group":"Outrights","variable":"Z24","pc":"PC3","loading":0.0843236758},{"group":"Outrights","variable":"H25","pc":"PC1","loading":-0.1421711287},{"group":"Outrights","variable":"H25","pc":"PC2","loading":0.2003857321},{"group":"Outrights","variable":"H25","pc":"PC3","loading":0.0856149463},{"group":"Outrights","variable":"M25","pc":"PC1","loading":-0.1312385472},{"group":"Outrights","variable":"M25","pc":"PC2","loading":0.2070966468},{"group":"Outrights","variable":"M25","pc":"PC3","loading":0.0866874015},{"group":"Outrights","variable":"U25","pc":"PC1","loading":-0.1206775586},{"group":"Outrights","variable":"U25","pc":"PC2","loading":0.2127814476},{"group":"Outrights","variable":"U25","pc":"PC3","loading":0.0875932543},{"group":"Outrights","variable":"Z25","pc":"PC1","loading":-0.1104698539},{"group":"Outrights","variable":"Z25","pc":"PC2","loading":0.2175027878},{"group":"Outrights","variable":"Z25","pc":"PC3","loading":0.0883658596},{"group":"Outrights","variable":"H26","pc":"PC1","loading":-0.1006001095},{"group":"Outrights","variable":"H26","pc":"PC2","loading":0.2213192083},{"group":"Outrights","variable":"H26","pc":"PC3","loading":0.0890333276},{"group":"Outrights","variable":"M26","pc":"PC1","loading":-0.0910548174},{"group":"Outrights","variable":"M26","pc":"PC2","loading":0.2242858348},{"group":"Outrights","variable":"M26","pc":"PC3","loading":0.089606825},{"group":"Outrights","variable":"U26","pc":"PC1","loading":-0.0818224765},{"group":"Outrights","variable":"U26","pc":"PC2","loading":0.226456079},{"group":"Outrights","variable":"U26","pc":"PC3","loading":0.0900953041},{"group":"Outrights","variable":"Z26","pc":"PC1","loading":-0.0728923984},{"group":"Outrights","variable":"Z26","pc":"PC2","loading":0.2278807953},{"group":"Outrights","variable":"Z26","pc":"PC3","loading":0.090507316},{"group":"Outrights","variable":"H27","pc":"PC1","loading":-0.0642533866},{"group":"Outrights","variable":"H27","pc":"PC2","loading":0.2286082987},{"group":"Outrights","variable":"H27","pc":"PC3","loading":0.0908518931},{"group":"Outrights","variable":"M27","pc":"PC1","loading":-0.0558957816},{"group":"Outrights","variable":"M27","pc":"PC2","loading":0.228639536},{"group":"Outrights","variable":"M27","pc":"PC3","loading":0.0911370213},{"group":"Outrights","variable":"U27","pc":"PC1","loading":-0.0478107931},{"group":"Outrights","variable":"U27","pc":"PC2","loading":0.2280205809},{"group":"Outrights","variable":"U27","pc":"PC3","loading":0.0913693259},{"group":"Outrights","variable":"Z27","pc":"PC1","loading":-0.0400000004},{"group":"Outrights","variable":"Z27","pc":"PC2","loading":0.2267923483},{"group":"Outrights","variable":"Z27","pc":"PC3","loading":0.0915555468},{"group":"Outrights","variable":"H28","pc":"PC1","loading":-0.0324545239},{"group":"Outrights","variable":"H28","pc":"PC2","loading":0.2249969476},{"group":"Outrights","variable":"H28","pc":"PC3","loading":0.091699933},{"group":"Outrights","variable":"M28","pc":"PC1","loading":-0.0251649963},{"group":"Outrights","variable":"M28","pc":"PC2","loading":0.2226848737},{"group":"Outrights","variable":"M28","pc":"PC3","loading":0.0918074911},{"group":"Outrights","variable":"U228","pc":"PC1","loading":-0.0181238478},{"group":"Outrights","variable":"U228","pc":"PC2","loading":0.2198941014},{"group":"Outrights","variable":"U228","pc":"PC3","loading":0.0918831968},{"group":"Outrights","variable":"Z228","pc":"PC1","loading":-0.0113222303},{"group":"Outrights","variable":"Z228","pc":"PC2","loading":0.2166779453},{"group":"Outrights","variable":"Z228","pc":"PC3","loading":0.0919313271},{"group":"Outrights","variable":"H29","pc":"PC1","loading":-0.0047514332},{"group":"Outrights","variable":"H29","pc":"PC2","loading":0.2130836511},{"group":"Outrights","variable":"H29","pc":"PC3","loading":0.0919567954},{"group":"Outrights","variable":"M29","pc":"PC1","loading":0.001684501},{"group":"Outrights","variable":"M29","pc":"PC2","loading":0.2091653859},{"group":"Outrights","variable":"M29","pc":"PC3","loading":0.0919639537},{"group":"Outrights","variable":"U29","pc":"PC1","loading":0.0079015949},{"group":"Outrights","variable":"U29","pc":"PC2","loading":0.2049817799},{"group":"Outrights","variable":"U29","pc":"PC3","loading":0.0919566991},{"group":"Outrights","variable":"Z29","pc":"PC1","loading":0.0138981442},{"group":"Outrights","variable":"Z29","pc":"PC2","loading":0.2005878496},{"group":"Outrights","variable":"Z29","pc":"PC3","loading":0.0919383091},{"group":"Fly Spreads","variable":"Z20 / H21 / M21","pc":"PC1","loading":0.0010041262},{"group":"Fly Spreads","variable":"Z20 / H21 / M21","pc":"PC2","loading":-0.0210255198},{"group":"Fly Spreads","variable":"Z20 / H21 / M21","pc":"PC3","loading":-0.0078761274},{"group":"Fly Spreads","variable":"H21 / M21 / U21","pc":"PC1","loading":0.0163351984},{"group":"Fly Spreads","variable":"H21 / M21 / U21","pc":"PC2","loading":-0.0078235212},{"group":"Fly Spreads","variable":"H21 / M21 / U21","pc":"PC3","loading":-0.0075306325},{"group":"Fly Spreads","variable":"M21 / U21 / H22","pc":"PC1","loading":0.029853383},{"group":"Fly Spreads","variable":"M21 / U21 / H22","pc":"PC2","loading":0.0045591526},{"group":"Fly Spreads","variable":"M21 / U21 / H22","pc":"PC3","loading":-0.0072704172},{"group":"Fly Spreads","variable":"U21 / H22 / M22","pc":"PC1","loading":0.0416973347},{"group":"Fly Spreads","variable":"U21 / H22 / M22","pc":"PC2","loading":0.0145293235},{"group":"Fly Spreads","variable":"U21 / H22 / M22","pc":"PC3","loading":-0.0070199738},{"group":"Fly Spreads","variable":"H22 / M22 / U22","pc":"PC1","loading":0.0518776092},{"group":"Fly Spreads","variable":"H22 / M22 / U22","pc":"PC2","loading":0.0217985474},{"group":"Fly Spreads","variable":"H22 / M22 / U22","pc":"PC3","loading":-0.0067868541},{"group":"Fly Spreads","variable":"M22 / U22 / Z21","pc":"PC1","loading":0.0573937968},{"group":"Fly Spreads","variable":"M22 / U22 / Z21","pc":"PC2","loading":0.023247076},{"group":"Fly Spreads","variable":"M22 / U22 / Z21","pc":"PC3","loading":-0.0070044573},{"group":"Fly Spreads","variable":"U22 / Z21 / Z22","pc":"PC1","loading":0.0578642732},{"group":"Fly Spreads","variable":"U22 / Z21 / Z22","pc":"PC2","loading":0.0236162386},{"group":"Fly Spreads","variable":"U22 / Z21 / Z22","pc":"PC3","loading":-0.0071337422},{"group":"Fly Spreads","variable":"Z21 / Z22 / H23","pc":"PC1","loading":0.0381655181},{"group":"Fly Spreads","variable":"Z21 / Z22 / H23","pc":"PC2","loading":-0.0094065662},{"group":"Fly Spreads","variable":"Z21 / Z22 / H23","pc":"PC3","loading":-0.0006248987},{"group":"Fly Spreads","variable":"Z22 / H23 / M23","pc":"PC1","loading":0.0305888062},{"group":"Fly Spreads","variable":"Z22 / H23 / M23","pc":"PC2","loading":-0.0099444155},{"group":"Fly Spreads","variable":"Z22 / H23 / M23","pc":"PC3","loading":-0.0006326177},{"group":"Fly Spreads","variable":"H23 / M23 / U23","pc":"PC1","loading":0.028795123},{"group":"Fly Spreads","variable":"H23 / M23 / U23","pc":"PC2","loading":-0.0105307524},{"group":"Fly Spreads","variable":"H23 / M23 / U23","pc":"PC3","loading":-0.0005500431},{"group":"Fly Spreads","variable":"M23 / U23 / Z23","pc":"PC1","loading":0.027063385},{"group":"Fly Spreads","variable":"M23 / U23 / Z23","pc":"PC2","loading":-0.0109960759},{"group":"Fly Spreads","variable":"M23 / U23 / Z23","pc":"PC3","loading":-0.0004907996},{"group":"Fly Spreads","variable":"U23 / Z23 / H24","pc":"PC1","loading":0.0253489893},{"group":"Fly Spreads","variable":"U23 / Z23 / H24","pc":"PC2","loading":-0.0113426767},{"group":"Fly Spreads","variable":"U23 / Z23 / H24","pc":"PC3","loading":-0.0004467554},{"group":"Fly Spreads","variable":"Z23 / H24 / M24","pc":"PC1","loading":0.0236467332},{"group":"Fly Spreads","variable":"Z23 / H24 / M24","pc":"PC2","loading":-0.0115814578},{"group":"Fly Spreads","variable":"Z23 / H24 / M24","pc":"PC3","loading":-0.0004117015},{"group":"Fly Spreads","variable":"H24 / M24 / U24","pc":"PC1","loading":0.0219602597},{"group":"Fly Spreads","variable":"H24 / M24 / U24","pc":"PC2","loading":-0.0117215234},{"group":"Fly Spreads","variable":"H24 / M24 / U24","pc":"PC3","loading":-0.0003823438},{"group":"Fly Spreads","variable":"M24 / U24 / Z24","pc":"PC1","loading":0.0202927233},{"group":"Fly Spreads","variable":"M24 / U24 / Z24","pc":"PC2","loading":-0.0117711422},{"group":"Fly Spreads","variable":"M24 / U24 / Z24","pc":"PC3","loading":-0.0003565983},{"group":"Fly Spreads","variable":"U24 / Z24 / H25","pc":"PC1","loading":0.0186469446},{"group":"Fly Spreads","variable":"U24 / Z24 / H25","pc":"PC2","loading":-0.0117366752},{"group":"Fly Spreads","variable":"U24 / Z24 / H25","pc":"PC3","loading":-0.0003332463},{"group":"Fly Spreads","variable":"Z24 / H25 / M25","pc":"PC1","loading":0.0170252655},{"group":"Fly Spreads","variable":"Z24 / H25 / M25","pc":"PC2","loading":-0.0116238692},{"group":"Fly Spreads","variable":"Z24 / H25 / M25","pc":"PC3","loading":-0.0003114407},{"group":"Fly Spreads","variable":"H25 / M25 / U25","pc":"PC1","loading":0.0154297151},{"group":"Fly Spreads","variable":"H25 / M25 / U25","pc":"PC2","loading":-0.011441853},{"group":"Fly Spreads","variable":"H25 / M25 / U25","pc":"PC3","loading":-0.0002905391},{"group":"Fly Spreads","variable":"M25 / U25 / Z25","pc":"PC1","loading":0.0138619632},{"group":"Fly Spreads","variable":"M25 / U25 / Z25","pc":"PC2","loading":-0.0111979331},{"group":"Fly Spreads","variable":"M25 / U25 / Z25","pc":"PC3","loading":-0.0002699252},{"group":"Fly Spreads","variable":"U25 / Z25 / H26","pc":"PC1","loading":0.0123233199},{"group":"Fly Spreads","variable":"U25 / Z25 / H26","pc":"PC2","loading":-0.0109012423},{"group":"Fly Spreads","variable":"U25 / Z25 / H26","pc":"PC3","loading":-0.0002490219},{"group":"Fly Spreads","variable":"Z25 / H26 / M26","pc":"PC1","loading":0.010815049},{"group":"Fly Spreads","variable":"Z25 / H26 / M26","pc":"PC2","loading":-0.0105607596},{"group":"Fly Spreads","variable":"Z25 / H26 / M26","pc":"PC3","loading":-0.0002272895},{"group":"Fly Spreads","variable":"H26 / M26 / U26","pc":"PC1","loading":0.009338304},{"group":"Fly Spreads","variable":"H26 / M26 / U26","pc":"PC2","loading":-0.0101850116},{"group":"Fly Spreads","variable":"H26 / M26 / U26","pc":"PC3","loading":-0.000204179},{"group":"Fly Spreads","variable":"M26 / U26 / Z26","pc":"PC1","loading":0.0078941795},{"group":"Fly Spreads","variable":"M26 / U26 / Z26","pc":"PC2","loading":-0.0097825296},{"group":"Fly Spreads","variable":"M26 / U26 / Z26","pc":"PC3","loading":-0.0001791834},{"group":"Fly Spreads","variable":"U26 / Z26 / H27","pc":"PC1","loading":0.0064837581},{"group":"Fly Spreads","variable":"U26 / Z26 / H27","pc":"PC2","loading":-0.0093616672},{"group":"Fly Spreads","variable":"U26 / Z26 / H27","pc":"PC3","loading":-0.0001518063},{"group":"Fly Spreads","variable":"Z26 / H27 / M27","pc":"PC1","loading":0.0051080447},{"group":"Fly Spreads","variable":"Z26 / H27 / M27","pc":"PC2","loading":-0.0089278438},{"group":"Fly Spreads","variable":"Z26 / H27 / M27","pc":"PC3","loading":-0.0001215456},{"group":"Fly Spreads","variable":"H27 / M27 / U27","pc":"PC1","loading":0.0037678508},{"group":"Fly Spreads","variable":"H27 / M27 / U27","pc":"PC2","loading":-0.0084865181},{"group":"Fly Spreads","variable":"H27 / M27 / U27","pc":"PC3","loading":-0.0000878174},{"group":"Fly Spreads","variable":"M27 / U27 / Z27","pc":"PC1","loading":0.0024638787},{"group":"Fly Spreads","variable":"M27 / U27 / Z27","pc":"PC2","loading":-0.0080436407},{"group":"Fly Spreads","variable":"M27 / U27 / Z27","pc":"PC3","loading":-0.0000500473},{"group":"Fly Spreads","variable":"U27 / Z27 / H28","pc":"PC1","loading":0.0011966579},{"group":"Fly Spreads","variable":"U27 / Z27 / H28","pc":"PC2","loading":-0.0076043444},{"group":"Fly Spreads","variable":"U27 / Z27 / H28","pc":"PC3","loading":-0.0000076899},{"group":"Fly Spreads","variable":"Z27 / H28 / M28","pc":"PC1","loading":-0.0009369947},{"group":"Fly Spreads","variable":"Z27 / H28 / M28","pc":"PC2","loading":-0.0071736697},{"group":"Fly Spreads","variable":"Z27 / H28 / M28","pc":"PC3","loading":0.0000388701},{"group":"Fly Spreads","variable":"H28 / M28 / U228","pc":"PC1","loading":-0.0025983804},{"group":"Fly Spreads","variable":"H28 / M28 / U228","pc":"PC2","loading":-0.006756855},{"group":"Fly Spreads","variable":"H28 / M28 / U228","pc":"PC3","loading":0.000090226},{"group":"Fly Spreads","variable":"M28 / U228 / Z228","pc":"PC1","loading":-0.0040332832},{"group":"Fly Spreads","variable":"M28 / U228 / Z228","pc":"PC2","loading":-0.0063581729},{"group":"Fly Spreads","variable":"M28 / U228 / Z228","pc":"PC3","loading":0.0001469032},{"group":"Fly Spreads","variable":"U228 / Z228 / H29","pc":"PC1","loading":-0.0052219082},{"group":"Fly Spreads","variable":"U228 / Z228 / H29","pc":"PC2","loading":-0.0059738096},{"group":"Fly Spreads","variable":"U228 / Z228 / H29","pc":"PC3","loading":0.0002092109},{"group":"Fly Spreads","variable":"Z228 / H29 / M29","pc":"PC1","loading":-0.0061406834},{"group":"Fly Spreads","variable":"Z228 / H29 / M29","pc":"PC2","loading":-0.0056108168},{"group":"Fly Spreads","variable":"Z228 / H29 / M29","pc":"PC3","loading":0.000277498},{"group":"Fly Spreads","variable":"H29 / M29 / U29","pc":"PC1","loading":-0.0067711466},{"group":"Fly Spreads","variable":"H29 / M29 / U29","pc":"PC2","loading":-0.0052737604},{"group":"Fly Spreads","variable":"H29 / M29 / U29","pc":"PC3","loading":0.0003522204},{"group":"Fly Spreads","variable":"M29 / U29 / Z29","pc":"PC1","loading":-0.0071068832},{"group":"Fly Spreads","variable":"M29 / U29 / Z29","pc":"PC2","loading":-0.0049692461},{"group":"Fly Spreads","variable":"M29 / U29 / Z29","pc":"PC3","loading":0.000433705}]
    const pcSummary = {"PC1":"PC1: 99.87% (Level Factor)","PC2":"PC2: 0.11% (Slope Factor)","PC3":"PC3: 0.01% (Curvature Factor)","Total":"Total Variance Explained: 99.99%"}
    // --- End of Injected Data ---

    function renderSummary() {
        const summaryDiv = d3.select("#pc-summary");
        
        Object.entries(pcSummary).forEach(([key, value]) => {
            summaryDiv.append("div")
                .attr("class", `p-3 rounded-lg text-sm font-semibold shadow-inner ${key === 'Total' ? 'bg-indigo-100 text-indigo-800' : 'bg-white text-gray-700 border border-gray-200'}`)
                .text(value);
        });
    }

    function renderHeatmap() {
        // Setup dimensions
        const margin = { top: 30, right: 20, bottom: 200, left: 100 },
              width = 1100 - margin.left - margin.right,
              height = 800 - margin.top - margin.bottom;

        // Get unique PC components and variables (contracts/spreads)
        const pcs = Array.from(new Set(heatmapData.map(d => d.pc)));
        const groups = Array.from(new Set(heatmapData.map(d => d.group)));
        
        // Ensure the order of variables matches the contract timeline + then flies
        const outrightVariables = Array.from(new Set(heatmapData.filter(d => d.group === 'Outrights').map(d => d.variable)));
        const flyVariables = Array.from(new Set(heatmapData.filter(d => d.group === 'Fly Spreads').map(d => d.variable)));
        const variables = outrightVariables.concat(flyVariables);

        // Append the svg object to the div
        const svg = d3.select("#heatmap")
            .append("svg")
            .attr("width", width + margin.left + margin.right)
            .attr("height", height + margin.top + margin.bottom)
            .append("g")
            .attr("transform", `translate(${margin.left},${margin.top})`);

        // X scale (PC components)
        const x = d3.scaleBand()
            .range([0, width])
            .domain(pcs)
            .padding(0.05);

        // Y scale (Outrights and Flies)
        const y = d3.scaleBand()
            .range([height, 0])
            .domain(variables)
            .padding(0.05);

        // Color scale (Loadings)
        const maxLoading = d3.max(heatmapData, d => Math.abs(d.loading));
        const colorScale = d3.scaleSequential(d3.interpolateRdBu)
            .domain([maxLoading, -maxLoading]); // Red (positive) to Blue (negative)

        // Tooltip
        const tooltip = d3.select("#app")
            .append("div")
            .attr("class", "tooltip opacity-0 transition-opacity duration-200")
            .style("opacity", 0);

        // Mouse handlers
        const mouseover = function(event, d) {
            tooltip.style("opacity", 1);
            d3.select(this).style("stroke", "#10b981").style("stroke-width", 2);
        }
        const mousemove = function(event, d) {
            const loadingText = (d.loading > 0 ? "+" : "") + d.loading.toFixed(4);
            tooltip
                .html(`**${d.variable}**<br>PC: ${d.pc}<br>Loading: ${loadingText}`)
                .style("left", (event.pageX + 10) + "px")
                .style("top", (event.pageY - 50) + "px");
        }
        const mouseleave = function(event, d) {
            tooltip.style("opacity", 0);
            d3.select(this).style("stroke", "none");
        }


        // Draw the heatmap rectangles
        svg.selectAll()
            .data(heatmapData, d => d.pc + ':' + d.variable)
            .enter()
            .append("rect")
            .attr("x", d => x(d.pc))
            .attr("y", d => y(d.variable))
            .attr("rx", 4)
            .attr("ry", 4)
            .attr("width", x.bandwidth())
            .attr("height", y.bandwidth())
            .style("fill", d => colorScale(d.loading))
            .on("mouseover", mouseover)
            .on("mousemove", mousemove)
            .on("mouseleave", mouseleave);

        // Add X axis (PC components)
        svg.append("g")
            .attr("class", "axis")
            .attr("transform", `translate(0, ${height})`)
            .call(d3.axisBottom(x))
            .selectAll("text")
            .attr("transform", "translate(0,10) rotate(-45)")
            .style("text-anchor", "end");

        // Add Y axis (Contracts/Spreads)
        svg.append("g")
            .attr("class", "axis")
            .call(d3.axisLeft(y).tickSize(0))
            .selectAll("text")
            .style("font-size", "11px");

        // Add X axis label
        svg.append("text")
            .attr("class", "text-lg font-medium text-gray-700")
            .attr("text-anchor", "middle")
            .attr("x", width / 2)
            .attr("y", height + margin.bottom - 40)
            .text("Principal Components (PCs)");

        // Add Y axis label (Rotated)
        svg.append("text")
            .attr("class", "text-lg font-medium text-gray-700")
            .attr("text-anchor", "middle")
            .attr("transform", "rotate(-90)")
            .attr("y", -margin.left + 20)
            .attr("x", -height / 2)
            .text("Financial Instruments");

        // Add a horizontal line to separate Outrights and Fly Spreads
        const flyStart = y(outrightVariables[outrightVariables.length - 1]) + y.bandwidth() + y.paddingInner() * y.bandwidth();
        svg.append("line")
            .attr("class", "group-separator")
            .attr("x1", 0)
            .attr("y1", flyStart)
            .attr("x2", width)
            .attr("y2", flyStart);

        // Add labels for groups
        svg.append("text")
            .attr("x", -margin.left + 5)
            .attr("y", 10)
            .style("font-weight", "bold")
            .text("Outrights");

        svg.append("text")
            .attr("x", -margin.left + 5)
            .attr("y", flyStart + 25)
            .style("font-weight", "bold")
            .text("Fly Spreads");
    }

    // Initialize the page
    window.onload = function() {
        renderSummary();
        renderHeatmap();
    };

</script>
</body>
</html>

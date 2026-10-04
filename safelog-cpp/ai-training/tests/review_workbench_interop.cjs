"use strict";
const fs = require("node:fs");
const api = require("../review-ui/review-workbench.js");
const data = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
const results = data.fixtures.map(item => {
  try {
    const bytes = Buffer.from(item.bytes_base64, "base64");
    const source = new TextDecoder("utf-8", {fatal: true, ignoreBOM: true}).decode(bytes);
    api.validateFeedback(api.parseStrictJSON(source), data.metadata);
    return {name: item.name, accepted: true};
  } catch (error) { return {name: item.name, accepted: false}; }
});
process.stdout.write(JSON.stringify(results));

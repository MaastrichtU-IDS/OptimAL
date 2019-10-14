class: CommandLineTool
cwlVersion: v1.0
$namespaces:
  sbg: 'https://www.sevenbridges.com/'
id: extract_xml
baseCommand:
  - python3
inputs:
  - id: coding
    type: string
  - id: xml_path
    type: string
  - id: abs_path
    type: string
outputs:
  - id: output
    type: File
    outputBinding:
        glob: XMLProduct.csv
label: extract-xml
arguments:
  - $(inputs.abs_path)src/DailymedXMLExtracter.py
  - '-c'
  - $(inputs.coding)
  - '-i'
  - $(inputs.abs_path)$(inputs.xml_path)
  - '-o'
  - $(runtime.outdir)/XMLProduct.csv
requirements:
  - class: InlineJavascriptRequirement

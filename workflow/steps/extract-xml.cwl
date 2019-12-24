class: CommandLineTool
cwlVersion: v1.0
$namespaces:
  sbg: 'https://www.sevenbridges.com/'
id: extract_xml
baseCommand:
  - python3
inputs:
  - id: coding
    type:
      - string
      - type: array
        items: string
  - id: xml_path
    type: Directory
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
  - $(inputs.xml_path.path)
  - '-o'
  - $(runtime.outdir)/XMLProduct.csv
requirements:
  - class: InlineJavascriptRequirement

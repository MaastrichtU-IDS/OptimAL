class: CommandLineTool
cwlVersion: v1.0
$namespaces:
  sbg: 'https://www.sevenbridges.com/'
id: clean_text
baseCommand: python3
inputs:
  - id: xml_product
    type: File
  - id: abs_path
    type: string
outputs:
  - id: output
    type: File
    outputBinding:
      glob: XMLProduct_cleaned.csv
label: clean-text
arguments:
  - $(inputs.abs_path)src/TextClean.py
  - '-in'
  - $(inputs.xml_product)
  - '-out'
  - $(runtime.outdir)/XMLProduct_cleaned.csv
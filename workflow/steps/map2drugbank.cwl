class: CommandLineTool
cwlVersion: v1.0
$namespaces:
  sbg: 'https://www.sevenbridges.com/'
id: map2drugbank
baseCommand:
  - python3
inputs:
  - id: drugbank_mapping
    type: string
  - id: xml_product_cleaned
    type: File
  - id: abs_path
    type: string
outputs:   
  - id: output
    type: File
    outputBinding:
      glob: XMLProduct_DBID.csv
label: map2DrugBank
arguments:
  - $(inputs.abs_path)src/DBIDmerge.py
  - '-m'
  - $(inputs.abs_path)$(inputs.drugbank_mapping)
  - '-i'
  - $(inputs.xml_product_cleaned)
  - '-o'
  - $(runtime.outdir)/XMLProduct_DBID.csv

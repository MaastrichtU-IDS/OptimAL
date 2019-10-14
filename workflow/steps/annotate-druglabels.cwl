class: CommandLineTool
cwlVersion: v1.0
$namespaces:
  sbg: 'https://www.sevenbridges.com/'
id: annotate_druglabels
baseCommand:
  - python3
inputs:
  - id: xml_product_dbid
    type: File
  - id: abs_path
    type: string

outputs: 
  - id: output
    type: File
    outputBinding:
      glob: XMLProduct_annotations.csv
label: annotate-druglabels
arguments:
  - $(inputs.abs_path)src/BPAnnotator.py
  - '-i'
  - $(inputs.xml_product_dbid)
  - '-o'
  - $(runtime.outdir)/XMLProduct_annotations.csv




python drugindication_ml/src/cv_test.py -g drugindication_ml/data/input/unified-gold-standard-umls.txt -dr drugindication_ml/data/features/drugs-targets.txt drugindication_ml/data/features/drugs-fingerprint.txt drugindication_ml/data/features/drugs-sider-se.txt -di drugindication_ml/data/features/diseases-ndfrt-meddra.txt -o drugindication_ml/data/output/completeset_unified_validation.txt -disjoint 0 -p 2 -m rf -fs drugindication_ml/data/output/selected_features.csv -nr 2 -nf 5


python drugindication_ml/src/train_and_test.py -g drugindication_ml/data/input/unified-gold-standard-umls.txt -t data/output/unlabeled_for_crowd.csv -dr drugindication_ml/data/features/drugs-fingerprint.txt drugindication_ml/data/features/drugs-targets.txt -di drugindication_ml/data/features/diseases-ndfrt-meddra.txt -m rf -p 2 -s 100 -o data/output/predictions_for_unlabeled.csv



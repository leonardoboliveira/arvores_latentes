-in
"D:\GDrive\Puc\Projeto Final\Datasets\conll\devel.conll"
-model
"D:\GDrive\Puc\Projeto Final\models\cort\model_only_embeding.obj"
-out
predict.encoded
-extractor
extension.antecedent_trees.extract_substructures_limited
-perceptron
cort.coreference.approaches.antecedent_trees.AntecedentTreePerceptron
-features
"D:\GDrive\Puc\Projeto Final\Code\extra_files\features.txt"
-instance_extractor
extension.instance_extractors.InstanceExtractor
-cost_function
cort.coreference.cost_functions.cost_based_on_consistency
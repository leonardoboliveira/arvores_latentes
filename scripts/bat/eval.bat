set PYTHONPATH=./third_party

..\third_party\cort\bin\cort-predict-conll -in "D:\GDrive\Puc\Projeto Final\Datasets\conll\conll-2012-test.conll" ^
                   -model "d:\GDrive\Puc\Projeto Final\models\cort\model.devel.encoded.head.obj" ^
                   -out "D:\GDrive\Puc\Projeto Final\Datasets\predicted\predict.encoded" ^
                   -extractor cort.coreference.approaches.antecedent_trees.extract_substructures ^
                   -perceptron extension.antecedent_trees.AntecedentTreePerceptron ^
                   -clusterer extension.clusterer.all_ante ^
                   -features "D:\GDrive\Puc\Projeto Final\Code\extra_files\features.txt" ^
                   -ante "D:\GDrive\Puc\Projeto Final\Datasets\predicted\predict.encoded.ante" ^
                   -gold "D:\GDrive\Puc\Projeto Final\Datasets\conll\conll-2012-test.conll" ^
                   -instance_extractor extension.instance_extractors.InstanceExtractor
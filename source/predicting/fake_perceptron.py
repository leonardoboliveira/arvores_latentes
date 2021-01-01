class TFPerceptron():
    def __init__(self, model_name):
        self.model_name = model_name

    def predict(self, substructures, arc_information):
        valid_arcs = []

        for structure in substructures:
            chosen_arcs = {}
            for arc in structure:
                embedding = arc_information[arc]
                # print(f"Embeddings:{embedding}:{arc[0].is_coreferent_with(arc[1])}")

                if (embedding[0] > 0) or (embedding[1] == 0):
                    continue
                current = chosen_arcs.get(arc[1], [0, None])
                if embedding[1] > current[0]:
                    chosen_arcs[arc[1]] = [embedding[1], arc]

            this_arcs = [x[1] for x in chosen_arcs.values() if x[1] is not None]
            valid_arcs.append(this_arcs)
            for arc in this_arcs:
                if not arc[0].is_coreferent_with(arc[1]):
                    print(f"False Positive {arc}")

        return valid_arcs, None, None

    @staticmethod
    def get_coref_labels():
        return ['+']

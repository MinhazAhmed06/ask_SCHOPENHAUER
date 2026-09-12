from pydantic import BaseModel


class EvalSample(BaseModel):
    id: str
    question: str
    expected_chapter: str
    expected_keywords: list[str]
    ground_truth_summary: str


BENCHMARK_DATASET: list[EvalSample] = [
    EvalSample(
        id="eval_01",
        question="What does Voltaire say about happiness?",
        expected_chapter="II. Personality, or What a Man Is",
        expected_keywords=["Voltaire", "happiness", "leisure", "philosophy", "wisdom"],
        ground_truth_summary="Schopenhauer cites Voltaire stating that happiness consists in peaceful contemplation and intellectual leisure.",
    ),
    EvalSample(
        id="eval_02",
        question="Why does Schopenhauer reject the moral condemnation of suicide?",
        expected_chapter="On Suicide",
        expected_keywords=["suicide", "cowardice", "twaddle", "right", "experiment"],
        ground_truth_summary="Schopenhauer dismisses the claim that suicide is cowardly as senseless twaddle, arguing that no one has greater right over anything than their own person, though calling it a clumsy experiment.",
    ),
    EvalSample(
        id="eval_03",
        question="How does Schopenhauer distinguish between what a man is versus what a man has?",
        expected_chapter="I. Division of the Subject",
        expected_keywords=["personality", "property", "wealth", "inward", "nature"],
        ground_truth_summary="What a man is (personality, health, intellect) constitutes his subjective nature and true happiness, whereas what a man has (property and wealth) is secondary and unstable.",
    ),
    EvalSample(
        id="eval_04",
        question="What is Schopenhauer's view on Pantheism?",
        expected_chapter="Some Words on Pantheism",
        expected_keywords=["Pantheism", "world", "God", "absurd", "illusions"],
        ground_truth_summary="Schopenhauer critiques Pantheism as merely renaming the world into God without explaining the presence of suffering and evil.",
    ),
    EvalSample(
        id="eval_05",
        question="What does Schopenhauer state regarding the indestructibility of our true nature by death?",
        expected_chapter="On the Doctrine of the Indestructibility of Our True Nature by Death",
        expected_keywords=["death", "nature", "indestructibility", "will", "phenomenon"],
        ground_truth_summary="Death destroys the individual phenomenon and intellect, but the underlying Will and metaphysical essence remain indestructible.",
    ),
]

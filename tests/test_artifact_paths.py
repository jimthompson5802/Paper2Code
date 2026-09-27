from pathlib import Path
import unittest

from codes.utils import task_artifact_filename


class TaskArtifactFilenameTests(unittest.TestCase):
    def test_nested_task_artifact_filename_is_flat(self):
        task_name = "configs/default.yaml"

        filename = task_artifact_filename(task_name, "_simple_analysis.txt")
        artifact_path = Path("analyzing_artifacts") / filename

        self.assertEqual(filename, "configs_default.yaml_simple_analysis.txt")
        self.assertEqual(artifact_path.parent, Path("analyzing_artifacts"))

    def test_task_artifact_filename_preserves_simple_task_names(self):
        filename = task_artifact_filename(
            "requirements.txt", "_simple_analysis_response.json"
        )

        self.assertEqual(
            filename, "requirements.txt_simple_analysis_response.json"
        )

    def test_task_artifact_filename_flattens_windows_separators(self):
        filename = task_artifact_filename(r"configs\default.yaml", "_coding.txt")

        self.assertEqual(filename, "configs_default.yaml_coding.txt")


if __name__ == "__main__":
    unittest.main()

import os
import json
import random
import shutil
import numpy as np
from services.similarity import process_submissions
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

class GroundTruthDataset:
    """
    Clase para crear y validar un dataset de ground truth para detección de plagio
    """
    def __init__(self, base_dir="ground_truth"):
        self.base_dir = base_dir
        self.categories = ["original", "plagiarized", "modified"]
        self.metadata_file = os.path.join(base_dir, "metadata.json")
        
        # Crear directorio base si no existe
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
    
    def create_dataset_structure(self):
        """
        Crea la estructura de directorios para el dataset
        """
        for category in self.categories:
            category_dir = os.path.join(self.base_dir, category)
            if not os.path.exists(category_dir):
                os.makedirs(category_dir)
        
        print(f"Estructura de directorios creada en {self.base_dir}")
    
    def add_original_code(self, file_path, language, description=""):
        """
        Añade código original al dataset
        """
        # Crear directorio para este ejemplo
        example_id = f"original_{len(os.listdir(os.path.join(self.base_dir, 'original'))) + 1}"
        example_dir = os.path.join(self.base_dir, "original", example_id)
        os.makedirs(example_dir, exist_ok=True)
        
        # Copiar archivo
        file_name = os.path.basename(file_path)
        dest_path = os.path.join(example_dir, file_name)
        shutil.copy2(file_path, dest_path)
        
        # Actualizar metadata
        self._update_metadata({
            "id": example_id,
            "category": "original",
            "language": language,
            "file": file_name,
            "description": description,
            "related_examples": []
        })
        
        print(f"Código original añadido como {example_id}")
        return example_id
    
    def add_plagiarized_code(self, file_path, original_id, language, plagiarism_type, description=""):
        """
        Añade código plagiado al dataset
        """
        # Crear directorio para este ejemplo
        example_id = f"plagiarized_{len(os.listdir(os.path.join(self.base_dir, 'plagiarized'))) + 1}"
        example_dir = os.path.join(self.base_dir, "plagiarized", example_id)
        os.makedirs(example_dir, exist_ok=True)
        
        # Copiar archivo
        file_name = os.path.basename(file_path)
        dest_path = os.path.join(example_dir, file_name)
        shutil.copy2(file_path, dest_path)
        
        # Actualizar metadata
        self._update_metadata({
            "id": example_id,
            "category": "plagiarized",
            "language": language,
            "file": file_name,
            "description": description,
            "plagiarism_type": plagiarism_type,
            "original_id": original_id,
            "related_examples": [original_id]
        })
        
        # Actualizar metadata del original
        self._update_related_example(original_id, example_id)
        
        print(f"Código plagiado añadido como {example_id}")
        return example_id
    
    def add_modified_code(self, file_path, original_id, language, modification_type, description=""):
        """
        Añade código modificado (no plagiado) al dataset
        """
        # Crear directorio para este ejemplo
        example_id = f"modified_{len(os.listdir(os.path.join(self.base_dir, 'modified'))) + 1}"
        example_dir = os.path.join(self.base_dir, "modified", example_id)
        os.makedirs(example_dir, exist_ok=True)
        
        # Copiar archivo
        file_name = os.path.basename(file_path)
        dest_path = os.path.join(example_dir, file_name)
        shutil.copy2(file_path, dest_path)
        
        # Actualizar metadata
        self._update_metadata({
            "id": example_id,
            "category": "modified",
            "language": language,
            "file": file_name,
            "description": description,
            "modification_type": modification_type,
            "original_id": original_id,
            "related_examples": [original_id]
        })
        
        # Actualizar metadata del original
        self._update_related_example(original_id, example_id)
        
        print(f"Código modificado añadido como {example_id}")
        return example_id
    
    def _update_metadata(self, example_data):
        """
        Actualiza el archivo de metadata
        """
        metadata = self._load_metadata()
        
        # Añadir o actualizar ejemplo
        metadata["examples"][example_data["id"]] = example_data
        
        # Guardar metadata
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
    
    def _update_related_example(self, example_id, related_id):
        """
        Actualiza la lista de ejemplos relacionados
        """
        metadata = self._load_metadata()
        
        if example_id in metadata["examples"]:
            if "related_examples" not in metadata["examples"][example_id]:
                metadata["examples"][example_id]["related_examples"] = []
            
            if related_id not in metadata["examples"][example_id]["related_examples"]:
                metadata["examples"][example_id]["related_examples"].append(related_id)
            
            # Guardar metadata
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)
    
    def _load_metadata(self):
        """
        Carga el archivo de metadata o crea uno nuevo
        """
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            return {
                "examples": {},
                "test_pairs": []
            }
    
    def create_test_pairs(self, num_pairs=50):
        """
        Crea pares de prueba para validación
        """
        metadata = self._load_metadata()
        examples = metadata["examples"]
        
        # Limpiar pares de prueba existentes
        metadata["test_pairs"] = []
        
        # Crear pares positivos (plagiados)
        positive_pairs = []
        for example_id, example_data in examples.items():
            if example_data["category"] == "plagiarized":
                original_id = example_data["original_id"]
                positive_pairs.append({
                    "example1": original_id,
                    "example2": example_id,
                    "is_plagiarism": True
                })
        
        # Crear pares negativos (no plagiados)
        negative_pairs = []
        originals = [ex_id for ex_id, ex_data in examples.items() if ex_data["category"] == "original"]
        modifieds = [ex_id for ex_id, ex_data in examples.items() if ex_data["category"] == "modified"]
        
        # Pares original-original
        for i in range(len(originals)):
            for j in range(i+1, len(originals)):
                negative_pairs.append({
                    "example1": originals[i],
                    "example2": originals[j],
                    "is_plagiarism": False
                })
        
        # Pares original-modificado
        for orig_id in originals:
            for mod_id in modifieds:
                if mod_id not in examples[orig_id].get("related_examples", []):
                    negative_pairs.append({
                        "example1": orig_id,
                        "example2": mod_id,
                        "is_plagiarism": False
                    })
        
        # Seleccionar pares aleatorios
        num_positive = min(num_pairs // 2, len(positive_pairs))
        num_negative = min(num_pairs - num_positive, len(negative_pairs))
        
        selected_positive = random.sample(positive_pairs, num_positive)
        selected_negative = random.sample(negative_pairs, num_negative)
        
        # Combinar y guardar
        metadata["test_pairs"] = selected_positive + selected_negative
        random.shuffle(metadata["test_pairs"])
        
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Creados {len(metadata['test_pairs'])} pares de prueba")
        return metadata["test_pairs"]
    
    async def validate_detector(self, similarity_threshold=0.7):
        """
        Valida el detector usando los pares de prueba
        """
        metadata = self._load_metadata()
        test_pairs = metadata["test_pairs"]
        
        if not test_pairs:
            print("No hay pares de prueba definidos. Use create_test_pairs() primero.")
            return
        
        results = []
        y_true = []
        y_pred = []
        
        for pair in test_pairs:
            example1 = metadata["examples"][pair["example1"]]
            example2 = metadata["examples"][pair["example2"]]
            
            # Obtener rutas de archivos
            file1_path = os.path.join(self.base_dir, example1["category"], example1["id"], example1["file"])
            file2_path = os.path.join(self.base_dir, example2["category"], example2["id"], example2["file"])
            
            # Crear directorios temporales para process_submissions
            temp_dir1 = os.path.join(self.base_dir, "temp", "submission1")
            temp_dir2 = os.path.join(self.base_dir, "temp", "submission2")
            
            os.makedirs(temp_dir1, exist_ok=True)
            os.makedirs(temp_dir2, exist_ok=True)
            
            # Copiar archivos a directorios temporales
            shutil.copy2(file1_path, os.path.join(temp_dir1, example1["file"]))
            shutil.copy2(file2_path, os.path.join(temp_dir2, example2["file"]))
            
            # Procesar con el detector
            detection_results = await process_submissions(
                [temp_dir1, temp_dir2],
                example1["language"],
                f"validation_{pair['example1']}_{pair['example2']}"
            )
            
            # Verificar si se detectó plagio
            is_detected_plagiarism = False
            if detection_results["similarity_results"]:
                max_similarity = detection_results["similarity_results"][0]["max_similarity"]
                is_detected_plagiarism = max_similarity >= similarity_threshold
            
            # Guardar resultados
            results.append({
                "pair": pair,
                "detection_result": is_detected_plagiarism,
                "ground_truth": pair["is_plagiarism"],
                "match": is_detected_plagiarism == pair["is_plagiarism"]
            })
            
            y_true.append(pair["is_plagiarism"])
            y_pred.append(is_detected_plagiarism)
            
            # Limpiar directorios temporales
            shutil.rmtree(temp_dir1)
            shutil.rmtree(temp_dir2)
        
        # Calcular métricas
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
        cm = confusion_matrix(y_true, y_pred)
        
        # Guardar resultados
        validation_results = {
            "threshold": similarity_threshold,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "confusion_matrix": cm.tolist(),
            "detailed_results": results
        }
        
        with open(os.path.join(self.base_dir, "validation_results.json"), 'w', encoding='utf-8') as f:
            json.dump(validation_results, f, indent=2)
        
        print(f"Validación completada. F1-score: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
        return validation_results
    
    def generate_ground_truth_report(self):
        """
        Genera un informe sobre el dataset de ground truth
        """
        metadata = self._load_metadata()
        examples = metadata["examples"]
        
        # Contar ejemplos por categoría
        category_counts = {cat: 0 for cat in self.categories}
        for example in examples.values():
            category_counts[example["category"]] += 1
        
        # Contar tipos de plagio
        plagiarism_types = {}
        for example in examples.values():
            if example["category"] == "plagiarized":
                plag_type = example.get("plagiarism_type", "unknown")
                plagiarism_types[plag_type] = plagiarism_types.get(plag_type, 0) + 1
        
        # Contar tipos de modificación
        modification_types = {}
        for example in examples.values():
            if example["category"] == "modified":
                mod_type = example.get("modification_type", "unknown")
                modification_types[mod_type] = modification_types.get(mod_type, 0) + 1
        
        # Contar lenguajes
        languages = {}
        for example in examples.values():
            lang = example.get("language", "unknown")
            languages[lang] = languages.get(lang, 0) + 1
        
        # Generar informe
        report = {
            "total_examples": len(examples),
            "category_counts": category_counts,
            "plagiarism_types": plagiarism_types,
            "modification_types": modification_types,
            "languages": languages,
            "test_pairs": len(metadata.get("test_pairs", []))
        }
        
        with open(os.path.join(self.base_dir, "dataset_report.json"), 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
        
        print(f"Informe del dataset generado en {os.path.join(self.base_dir, 'dataset_report.json')}")
        return report

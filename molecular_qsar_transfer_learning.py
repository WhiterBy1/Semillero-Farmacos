"""
Sistema QSAR con Deep Learning usando Transfer Learning
Prediccion de actividad antiinflamatoria (inhibicion 5-LOX) a partir de multiples vistas 3D de moleculas

Arquitectura: VGG16 o ResNet50 con transfer learning
Estrategia de 2 fases:
  1. Entrenar con capas convolucionales congeladas
  2. Fine-tuning de todo el modelo
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from pathlib import Path
import json


class MolecularDataLoader:
    """
    Cargador de datos para moleculas con multiples vistas
    Compatible con estructura: Active_Result/Inactive_Result
    """
    
    def __init__(self, data_dir, img_size=(224, 224), num_views=6):
        """
        Args:
            data_dir: Directorio raiz que contiene Active_Result e Inactive_Result
            img_size: Tamano de las imagenes (height, width)
            num_views: Numero de vistas por molecula (6: 1D, 2D, 3D, 4D, 5D, 6D)
        """
        self.data_dir = Path(data_dir)
        self.img_size = img_size
        self.num_views = num_views
        self.molecules_data = []
        
    def scan_molecules(self):
        """
        Escanea las carpetas Active_Result e Inactive_Result
        y construye la lista de moleculas con sus etiquetas
        """
        print("Escaneando moleculas...")
        
        # Carpeta de moleculas activas
        active_dir = self.data_dir / 'Active_Result'
        if active_dir.exists():
            for mol_folder in active_dir.iterdir():
                if mol_folder.is_dir():
                    self.molecules_data.append({
                        'molecule_name': mol_folder.name,
                        'path': mol_folder,
                        'label': 1,  # Activa
                        'class_name': 'active'
                    })
        
        # Carpeta de moleculas inactivas
        inactive_dir = self.data_dir / 'Inactive_Result'
        if inactive_dir.exists():
            for mol_folder in inactive_dir.iterdir():
                if mol_folder.is_dir():
                    self.molecules_data.append({
                        'molecule_name': mol_folder.name,
                        'path': mol_folder,
                        'label': 0,  # Inactiva
                        'class_name': 'inactive'
                    })
        
        print(f"[OK] {len(self.molecules_data)} moleculas encontradas")
        
        # Mostrar distribucion
        active_count = sum(1 for m in self.molecules_data if m['label'] == 1)
        inactive_count = sum(1 for m in self.molecules_data if m['label'] == 0)
        print(f"  - Activas: {active_count}")
        print(f"  - Inactivas: {inactive_count}")
        
        return self.molecules_data
    
    def load_molecule_views(self, mol_path):
        """
        Carga las 6 vistas de una molecula
        
        Args:
            mol_path: Path a la carpeta de la molecula
            
        Returns:
            Lista de arrays numpy con las 6 vistas
        """
        views = []
        view_names = ['1D.png', '2D.png', '3D.png', '4D.png', '5D.png', '6D.png']
        
        for view_name in view_names:
            img_path = mol_path / view_name
            
            if not img_path.exists():
                # Si falta una vista, intentar sin extension o con otros formatos
                for ext in ['.jpg', '.jpeg', '.PNG', '.JPG']:
                    alt_path = mol_path / (view_name.replace('.png', ext))
                    if alt_path.exists():
                        img_path = alt_path
                        break
            
            if img_path.exists():
                try:
                    img = load_img(img_path, target_size=self.img_size)
                    img_array = img_to_array(img)
                    # Normalizar a [0, 1]
                    img_array = img_array / 255.0
                    views.append(img_array)
                except Exception as e:
                    print(f"Error cargando {img_path}: {e}")
                    # Crear imagen en blanco si falla
                    views.append(np.zeros((*self.img_size, 3)))
            else:
                print(f"Advertencia: {img_path} no encontrado")
                # Crear imagen en blanco
                views.append(np.zeros((*self.img_size, 3)))
        
        return views
    
    def create_dataset(self, molecules_list, batch_size=16, shuffle=True):
        """
        Crea un tf.data.Dataset para entrenamiento/validacion/test
        
        Args:
            molecules_list: Lista de diccionarios con informacion de moleculas
            batch_size: Tamano del batch
            shuffle: Si se debe mezclar los datos
            
        Returns:
            tf.data.Dataset
        """
        def generator():
            indices = list(range(len(molecules_list)))
            if shuffle:
                np.random.shuffle(indices)
            
            for idx in indices:
                mol = molecules_list[idx]
                views = self.load_molecule_views(mol['path'])
                label = mol['label']
                
                # Retornar (tuple_de_vistas, label) para compatibilidad con Keras
                yield (tuple(views), label)
        
        # Definir la signatura de salida
        # Input: tuple de 6 imagenes, Output: label
        output_signature = (
            tuple([
                tf.TensorSpec(shape=(*self.img_size, 3), dtype=tf.float32)
                for _ in range(self.num_views)
            ]),
            tf.TensorSpec(shape=(), dtype=tf.int32)
        )
        
        dataset = tf.data.Dataset.from_generator(
            generator,
            output_signature=output_signature
        )
        
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(molecules_list))
        
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    def train_val_test_split(self, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_state=42):
        """
        Divide los datos en train, validation y test
        """
        if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
            raise ValueError("Las proporciones deben sumar 1.0")
        
        # Primero dividir en train y temp
        train_mols, temp_mols = train_test_split(
            self.molecules_data,
            train_size=train_ratio,
            random_state=random_state,
            stratify=[m['label'] for m in self.molecules_data]
        )
        
        # Luego dividir temp en val y test
        val_size = val_ratio / (val_ratio + test_ratio)
        val_mols, test_mols = train_test_split(
            temp_mols,
            train_size=val_size,
            random_state=random_state,
            stratify=[m['label'] for m in temp_mols]
        )
        
        print(f"\n[OK] Division de datos:")
        print(f"  - Train: {len(train_mols)} moleculas ({train_ratio*100:.0f}%)")
        print(f"  - Validation: {len(val_mols)} moleculas ({val_ratio*100:.0f}%)")
        print(f"  - Test: {len(test_mols)} moleculas ({test_ratio*100:.0f}%)")
        
        return train_mols, val_mols, test_mols


class MolecularQSARModel:
    """
    Modelo QSAR basado en Transfer Learning para clasificacion binaria
    """
    
    def __init__(self, base_model_name='VGG16', input_shape=(224, 224, 3), num_views=6):
        """
        Args:
            base_model_name: 'VGG16' o 'ResNet50'
            input_shape: dimensiones de entrada (height, width, channels)
            num_views: numero de vistas por molecula
        """
        self.base_model_name = base_model_name
        self.input_shape = input_shape
        self.num_views = num_views
        self.model = None
        self.history = None
    
    def create_base_model(self):
        """Crear el modelo base pre-entrenado"""
        if self.base_model_name == 'VGG16':
            base_model = VGG16(
                weights='imagenet',
                include_top=False,
                input_shape=self.input_shape
            )
        elif self.base_model_name == 'ResNet50':
            base_model = ResNet50(
                weights='imagenet',
                include_top=False,
                input_shape=self.input_shape
            )
        else:
            raise ValueError("base_model_name debe ser 'VGG16' o 'ResNet50'")
        
        return base_model
    
    def build_multi_view_model(self, freeze_base=True):
        """
        Construir modelo que procesa multiples vistas de una molecula
        
        Args:
            freeze_base: Si True, congela las capas convolucionales
        """
        # Modelo base compartido
        base_model = self.create_base_model()
        
        if freeze_base:
            base_model.trainable = False
            print(f"[OK] Capas convolucionales de {self.base_model_name} congeladas")
        else:
            base_model.trainable = True
            print(f"[OK] Todas las capas de {self.base_model_name} descongeladas")
        
        # Inputs para cada vista
        view_inputs = []
        view_features = []
        
        for i in range(self.num_views):
            view_input = keras.Input(shape=self.input_shape, name=f'view_{i+1}')
            view_inputs.append(view_input)
            
            # Procesar cada vista
            x = base_model(view_input, training=not freeze_base)
            x = layers.GlobalAveragePooling2D()(x)
            view_features.append(x)
        
        # Concatenar features de todas las vistas
        if self.num_views > 1:
            combined = layers.Concatenate(name='concatenate_views')(view_features)
        else:
            combined = view_features[0]
        
        # Capas fully connected para fusion
        x = layers.Dense(512, activation='relu', name='fusion_fc1')(combined)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        
        x = layers.Dense(256, activation='relu', name='fusion_fc2')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        # Capa de salida para clasificacion binaria
        outputs = layers.Dense(1, activation='sigmoid', name='output')(x)
        
        model = keras.Model(
            inputs=view_inputs,
            outputs=outputs,
            name=f'{self.base_model_name}_MultiView_QSAR'
        )
        
        return model
    
    def compile_model(self, model, learning_rate=1e-4):
        """Compilar el modelo"""
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                tf.keras.metrics.AUC(name='auc'),
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall')
            ]
        )
        
        print(f"[OK] Modelo compilado con learning_rate={learning_rate}")
        return model
    
    def train(self, model, train_data, val_data, epochs=20, batch_size=16, 
              learning_rate=1e-3, checkpoint_path='best_model.keras'):
        """
        Entrenar el modelo
        
        Args:
            model: Modelo de Keras
            train_data: Dataset de entrenamiento
            val_data: Dataset de validacion
            epochs: Numero de epocas
            batch_size: Tamano del batch
            learning_rate: Tasa de aprendizaje
            checkpoint_path: Ruta para guardar el mejor modelo
        """
        # Compilar con learning rate especificado
        self.compile_model(model, learning_rate)
        
        # Callbacks
        callbacks = [
            ModelCheckpoint(
                checkpoint_path,
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        # Entrenar
        print(f"\nIniciando entrenamiento por {epochs} epocas...")
        history = model.fit(
            train_data,
            validation_data=val_data,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def plot_training_history(self, history_phase1, history_phase2=None, save_path='training_history.png'):
        """
        Visualizar el historial de entrenamiento
        
        Args:
            history_phase1: Historia de la fase 1
            history_phase2: Historia de la fase 2 (opcional)
            save_path: Ruta para guardar la figura
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Combinar historias si hay dos fases
        if history_phase2 is not None:
            # Fase 1
            epochs1 = len(history_phase1.history['loss'])
            # Fase 2
            epochs2 = len(history_phase2.history['loss'])
            total_epochs = epochs1 + epochs2
            
            # Combinar metricas
            train_loss = history_phase1.history['loss'] + history_phase2.history['loss']
            val_loss = history_phase1.history['val_loss'] + history_phase2.history['val_loss']
            train_acc = history_phase1.history['accuracy'] + history_phase2.history['accuracy']
            val_acc = history_phase1.history['val_accuracy'] + history_phase2.history['val_accuracy']
            
            # Linea divisoria entre fases
            phase_split = epochs1
        else:
            train_loss = history_phase1.history['loss']
            val_loss = history_phase1.history['val_loss']
            train_acc = history_phase1.history['accuracy']
            val_acc = history_phase1.history['val_accuracy']
            phase_split = None
        
        epochs = range(1, len(train_loss) + 1)
        
        # Loss
        axes[0, 0].plot(epochs, train_loss, 'b-', label='Train Loss')
        axes[0, 0].plot(epochs, val_loss, 'r-', label='Val Loss')
        if phase_split:
            axes[0, 0].axvline(x=phase_split, color='g', linestyle='--', label='Fine-tuning Start')
        axes[0, 0].set_title('Loss durante el entrenamiento')
        axes[0, 0].set_xlabel('Epoca')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy
        axes[0, 1].plot(epochs, train_acc, 'b-', label='Train Accuracy')
        axes[0, 1].plot(epochs, val_acc, 'r-', label='Val Accuracy')
        if phase_split:
            axes[0, 1].axvline(x=phase_split, color='g', linestyle='--', label='Fine-tuning Start')
        axes[0, 1].set_title('Accuracy durante el entrenamiento')
        axes[0, 1].set_xlabel('Epoca')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # AUC si esta disponible
        if 'auc' in history_phase1.history:
            if history_phase2 is not None:
                train_auc = history_phase1.history['auc'] + history_phase2.history['auc']
                val_auc = history_phase1.history['val_auc'] + history_phase2.history['val_auc']
            else:
                train_auc = history_phase1.history['auc']
                val_auc = history_phase1.history['val_auc']
            
            axes[1, 0].plot(epochs, train_auc, 'b-', label='Train AUC')
            axes[1, 0].plot(epochs, val_auc, 'r-', label='Val AUC')
            if phase_split:
                axes[1, 0].axvline(x=phase_split, color='g', linestyle='--', label='Fine-tuning Start')
            axes[1, 0].set_title('AUC durante el entrenamiento')
            axes[1, 0].set_xlabel('Epoca')
            axes[1, 0].set_ylabel('AUC')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        # Resumen final
        axes[1, 1].axis('off')
        summary_text = f"""
        Resumen del Entrenamiento:
        
        {'='*40}
        Fase 1 (Capas congeladas):
        - Epocas: {len(history_phase1.history['loss'])}
        - Loss final: {history_phase1.history['val_loss'][-1]:.4f}
        - Accuracy final: {history_phase1.history['val_accuracy'][-1]:.4f}
        """
        
        if history_phase2 is not None:
            summary_text += f"""
        {'='*40}
        Fase 2 (Fine-tuning):
        - Epocas: {len(history_phase2.history['loss'])}
        - Loss final: {history_phase2.history['val_loss'][-1]:.4f}
        - Accuracy final: {history_phase2.history['val_accuracy'][-1]:.4f}
            """
        
        axes[1, 1].text(0.1, 0.5, summary_text, fontsize=10, verticalalignment='center',
                       family='monospace')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[OK] Graficas guardadas en: {save_path}")
        plt.close()


def main_training_pipeline(data_dir, base_model='VGG16', num_views=6, batch_size=8):
    """
    Pipeline completo de entrenamiento con transfer learning en 2 fases
    
    Args:
        data_dir: Directorio raiz con Active_Result e Inactive_Result
        base_model: 'VGG16' o 'ResNet50'
        num_views: Numero de vistas por molecula
        batch_size: Tamano del batch
    """
    print("\n" + "="*70)
    print(f"PIPELINE DE TRANSFER LEARNING PARA QSAR - Modelo: {base_model}")
    print("="*70 + "\n")
    
    # 1. Cargar datos
    print("PASO 1: Cargando datos moleculares...")
    data_loader = MolecularDataLoader(
        data_dir=data_dir,
        img_size=(224, 224),
        num_views=num_views
    )
    
    data_loader.scan_molecules()
    
    # 2. Dividir datos
    print("\nPASO 2: Dividiendo datos en train/val/test...")
    train_mols, val_mols, test_mols = data_loader.train_val_test_split(
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15
    )
    
    # 3. Crear datasets
    print("\nPASO 3: Creando TensorFlow datasets...")
    train_dataset = data_loader.create_dataset(train_mols, batch_size=batch_size, shuffle=True)
    val_dataset = data_loader.create_dataset(val_mols, batch_size=batch_size, shuffle=False)
    test_dataset = data_loader.create_dataset(test_mols, batch_size=batch_size, shuffle=False)
    
    # 4. Crear modelo
    print("\nPASO 4: Construyendo modelo...")
    qsar_model = MolecularQSARModel(
        base_model_name=base_model,
        input_shape=(224, 224, 3),
        num_views=num_views
    )
    
    # Construir arquitectura multi-view
    model = qsar_model.build_multi_view_model(freeze_base=True)
    print("\n" + "="*70)
    print("ARQUITECTURA DEL MODELO:")
    print("="*70)
    model.summary()
    
    # 5. FASE 1: Entrenar con base congelada
    print("\n" + "="*70)
    print("FASE 1: Capas Convolucionales CONGELADAS")
    print("="*70)
    
    history_phase1 = qsar_model.train(
        model=model,
        train_data=train_dataset,
        val_data=val_dataset,
        epochs=20,
        batch_size=batch_size,
        learning_rate=1e-3,
        checkpoint_path='qsar_model_phase1_best.keras'
    )
    
    # Guardar modelo despues de fase 1
    model.save('qsar_model_phase1_final.keras')
    print("[OK] Modelo Fase 1 guardado")
    
    # 6. FASE 2: Fine-tuning
    print("\n" + "="*70)
    print("FASE 2: FINE-TUNING (descongelando todas las capas)")
    print("="*70)
    
    # Descongelar el modelo base
    for layer in model.layers:
        if 'vgg16' in layer.name.lower() or 'resnet50' in layer.name.lower():
            layer.trainable = True
    
    history_phase2 = qsar_model.train(
        model=model,
        train_data=train_dataset,
        val_data=val_dataset,
        epochs=30,
        batch_size=batch_size,
        learning_rate=1e-5,
        checkpoint_path='qsar_model_phase2_best.keras'
    )
    
    # 7. Guardar modelo final
    print("\nPASO 7: Guardando modelo final...")
    model.save('qsar_model_final.keras')
    print("[OK] Modelo final guardado")
    
    # 8. Visualizar resultados
    print("\nPASO 8: Generando visualizaciones...")
    qsar_model.plot_training_history(history_phase1, history_phase2, 
                                     save_path='training_history.png')
    
    # 9. Evaluar en test set
    print("\n" + "="*70)
    print("EVALUACION EN CONJUNTO DE PRUEBA")
    print("="*70)
    
    test_results = model.evaluate(test_dataset, verbose=1)
    
    print("\nRESULTADOS FINALES:")
    for metric_name, metric_value in zip(model.metrics_names, test_results):
        print(f"  {metric_name}: {metric_value:.4f}")
    
    print("\n[OK] Pipeline completado exitosamente!")
    
    return model, history_phase1, history_phase2, test_results


if __name__ == "__main__":
    print("="*70)
    print("Sistema QSAR con Transfer Learning")
    print("Prediccion de actividad antiinflamatoria (inhibicion 5-LOX)")
    print("usando multiples vistas 3D de moleculas")
    print("="*70 + "\n")
    
    # CONFIGURACION - AJUSTA ESTOS PARAMETROS
    data_directory = ''  # Carpeta que contiene Active_Result e Inactive_Result
    
    # Ejemplo: si tus datos estan en 'C:/Users/Usuario/datos_moleculares'
    # data_directory = 'C:/Users/Usuario/datos_moleculares'
    
    # Ejecutar pipeline
    try:
        model, hist1, hist2, test_results = main_training_pipeline(
            data_dir=data_directory,
            base_model='VGG16',  # Puedes cambiar a 'ResNet50'
            num_views=6,
            batch_size=8  # Ajustar segun memoria disponible
        )
        
        print("\n" + "="*70)
        print("ARCHIVOS GENERADOS:")
        print("="*70)
        print("  - qsar_model_phase1_best.keras: Mejor modelo de Fase 1")
        print("  - qsar_model_phase1_final.keras: Modelo final de Fase 1")
        print("  - qsar_model_phase2_best.keras: Mejor modelo de Fase 2")
        print("  - qsar_model_final.keras: Modelo final despues de Fine-tuning")
        print("  - training_history.png: Graficas del entrenamiento")
        
    except Exception as e:
        print(f"\n[ERROR] Error durante la ejecucion: {e}")
        print("\nVerifica que:")
        print("  1. La ruta data_directory sea correcta")
        print("  2. Existan las carpetas Active_Result e Inactive_Result")
        print("  3. Cada molecula tenga sus 6 vistas (1D.png - 6D.png)")
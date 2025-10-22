"""
Sistema QSAR con Deep Learning usando Transfer Learning
Predicción de actividad antiinflamatoria (inhibición 5-LOX) 
a partir de múltiples vistas 3D de moléculas

Arquitectura: VGG16 o ResNet50 con transfer learning
Estrategia: 
1. Congelar capas convolucionales
2. Entrenar nuevas capas fully connected
3. Fine-tuning de todo el modelo
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from pathlib import Path
import json


class MolecularQSARModel:
    """
    Modelo QSAR basado en Transfer Learning para predicción de actividad antiinflamatoria
    """
    
    def __init__(self, 
                 base_model_name='VGG16',
                 input_shape=(224, 224, 3),
                 num_views=6,
                 task='regression',
                 num_classes=1):
        """
        Args:
            base_model_name: 'VGG16' o 'ResNet50'
            input_shape: dimensiones de entrada (height, width, channels)
            num_views: número de vistas por molécula (6 en este caso)
            task: 'regression' para IC50/pIC50 o 'classification' para activo/inactivo
            num_classes: 1 para regresión, >1 para clasificación multiclase
        """
        self.base_model_name = base_model_name
        self.input_shape = input_shape
        self.num_views = num_views
        self.task = task
        self.num_classes = num_classes
        self.model = None
        self.history = None
        
    def create_base_model(self):
        """Crear el modelo base pre-entrenado (VGG16 o ResNet50)"""
        
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
    
    def build_single_view_model(self, freeze_base=True):
        """
        Construir modelo para una sola vista
        
        Args:
            freeze_base: Si True, congela las capas convolucionales del modelo base
        """
        
        # Modelo base pre-entrenado
        base_model = self.create_base_model()
        
        # FASE 1: Congelar capas convolucionales
        if freeze_base:
            base_model.trainable = False
            print(f"✓ Capas convolucionales de {self.base_model_name} congeladas")
        else:
            base_model.trainable = True
            print(f"✓ Todas las capas de {self.base_model_name} descongeladas para fine-tuning")
        
        # Construir el modelo completo
        inputs = keras.Input(shape=self.input_shape)
        
        # Pasar por el modelo base
        x = base_model(inputs, training=False if freeze_base else True)
        
        # Global Average Pooling
        x = layers.GlobalAveragePooling2D()(x)
        
        # Nuevas capas fully connected (parte lineal personalizada)
        x = layers.Dense(512, activation='relu', name='fc1')(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(256, activation='relu', name='fc2')(x)
        x = layers.Dropout(0.3)(x)
        
        # Capa de salida según la tarea
        if self.task == 'regression':
            outputs = layers.Dense(1, activation='linear', name='output')(x)
        else:
            activation = 'sigmoid' if self.num_classes == 1 else 'softmax'
            outputs = layers.Dense(self.num_classes, activation=activation, name='output')(x)
        
        model = keras.Model(inputs, outputs, name=f'{self.base_model_name}_QSAR')
        
        return model
    
    def build_multi_view_model(self, freeze_base=True):
        """
        Construir modelo que procesa múltiples vistas de una molécula
        Combina información de 6 vistas diferentes
        
        Args:
            freeze_base: Si True, congela las capas convolucionales
        """
        
        # Modelo base compartido para todas las vistas
        base_model = self.create_base_model()
        
        if freeze_base:
            base_model.trainable = False
            print(f"✓ Capas convolucionales de {self.base_model_name} congeladas")
        else:
            base_model.trainable = True
            print(f"✓ Todas las capas de {self.base_model_name} descongeladas")
        
        # Inputs para cada vista
        view_inputs = []
        view_features = []
        
        for i in range(self.num_views):
            # Input para cada vista
            view_input = keras.Input(shape=self.input_shape, name=f'view_{i+1}')
            view_inputs.append(view_input)
            
            # Procesar cada vista con el modelo base compartido
            x = base_model(view_input, training=False if freeze_base else True)
            x = layers.GlobalAveragePooling2D()(x)
            view_features.append(x)
        
        # Concatenar features de todas las vistas
        if self.num_views > 1:
            combined = layers.Concatenate(name='concatenate_views')(view_features)
        else:
            combined = view_features[0]
        
        # Nuevas capas fully connected para la fusión de vistas
        x = layers.Dense(1024, activation='relu', name='fusion_fc1')(combined)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(512, activation='relu', name='fusion_fc2')(x)
        x = layers.Dropout(0.4)(x)
        x = layers.Dense(256, activation='relu', name='fusion_fc3')(x)
        x = layers.Dropout(0.3)(x)
        
        # Capa de salida
        if self.task == 'regression':
            outputs = layers.Dense(1, activation='linear', name='output')(x)
        else:
            activation = 'sigmoid' if self.num_classes == 1 else 'softmax'
            outputs = layers.Dense(self.num_classes, activation=activation, name='output')(x)
        
        model = keras.Model(inputs=view_inputs, outputs=outputs, 
                          name=f'{self.base_model_name}_MultiView_QSAR')
        
        return model
    
    def compile_model(self, model, learning_rate=1e-4):
        """
        Compilar el modelo con optimizador y función de pérdida apropiados
        
        Args:
            model: modelo de Keras
            learning_rate: tasa de aprendizaje
        """
        
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        
        if self.task == 'regression':
            loss = 'mse'
            metrics = ['mae', 'mse', tf.keras.metrics.RootMeanSquaredError(name='rmse')]
        else:
            if self.num_classes == 1:
                loss = 'binary_crossentropy'
                metrics = ['accuracy', tf.keras.metrics.AUC(name='auc')]
            else:
                loss = 'categorical_crossentropy'
                metrics = ['accuracy']
        
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        
        print(f"✓ Modelo compilado con {loss} y learning_rate={learning_rate}")
        
        return model
    
    def train_phase1_frozen_base(self, 
                                  model,
                                  train_data, 
                                  val_data,
                                  epochs=20,
                                  batch_size=32,
                                  learning_rate=1e-3):
        """
        FASE 1: Entrenar solo las nuevas capas FC con base congelada
        
        Args:
            model: modelo de Keras
            train_data: datos de entrenamiento
            val_data: datos de validación
            epochs: número de épocas
            batch_size: tamaño del batch
            learning_rate: tasa de aprendizaje (mayor para fase 1)
        """
        
        print("\n" + "="*70)
        print("FASE 1: Entrenamiento con capas convolucionales CONGELADAS")
        print("="*70)
        
        # Compilar con learning rate más alto
        model = self.compile_model(model, learning_rate=learning_rate)
        
        # Callbacks
        callbacks = self._get_callbacks('phase1', monitor='val_loss')
        
        # Entrenar
        history1 = model.fit(
            train_data,
            validation_data=val_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        print(f"✓ Fase 1 completada - Mejor val_loss: {min(history1.history['val_loss']):.4f}")
        
        return history1
    
    def train_phase2_fine_tuning(self,
                                  model,
                                  train_data,
                                  val_data,
                                  epochs=30,
                                  batch_size=32,
                                  learning_rate=1e-5,
                                  unfreeze_from_layer=None):
        """
        FASE 2: Fine-tuning descongelando las capas convolucionales
        
        Args:
            model: modelo de Keras ya entrenado en fase 1
            train_data: datos de entrenamiento
            val_data: datos de validación
            epochs: número de épocas
            batch_size: tamaño del batch
            learning_rate: tasa de aprendizaje (menor para fine-tuning)
            unfreeze_from_layer: nombre o índice de capa desde donde descongelar
        """
        
        print("\n" + "="*70)
        print("FASE 2: FINE-TUNING con capas convolucionales DESCONGELADAS")
        print("="*70)
        
        # Descongelar el modelo base
        base_model = model.layers[1] if hasattr(model.layers[1], 'trainable') else model.layers[0]
        base_model.trainable = True
        
        # Opcionalmente, descongelar solo desde una capa específica
        if unfreeze_from_layer is not None:
            for layer in base_model.layers[:unfreeze_from_layer]:
                layer.trainable = False
            print(f"✓ Capas descongeladas desde: {unfreeze_from_layer}")
        else:
            print("✓ Todas las capas del modelo base descongeladas")
        
        # Recompilar con learning rate muy bajo
        model = self.compile_model(model, learning_rate=learning_rate)
        
        # Callbacks
        callbacks = self._get_callbacks('phase2', monitor='val_loss')
        
        # Entrenar
        history2 = model.fit(
            train_data,
            validation_data=val_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        print(f"✓ Fase 2 completada - Mejor val_loss: {min(history2.history['val_loss']):.4f}")
        
        return history2
    
    def _get_callbacks(self, phase_name, monitor='val_loss'):
        """Obtener callbacks para entrenamiento"""
        
        callbacks = [
            ModelCheckpoint(
                f'best_model_{phase_name}.keras',
                monitor=monitor,
                save_best_only=True,
                mode='min' if 'loss' in monitor else 'max',
                verbose=1
            ),
            EarlyStopping(
                monitor=monitor,
                patience=10 if phase_name == 'phase1' else 15,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor=monitor,
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        return callbacks
    
    def plot_training_history(self, history_phase1, history_phase2=None, save_path='training_history.png'):
        """Visualizar el historial de entrenamiento"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Determinar la métrica principal
        metric_name = 'mae' if self.task == 'regression' else 'accuracy'
        
        # Fase 1
        if history_phase1:
            epochs1 = range(1, len(history_phase1.history['loss']) + 1)
            
            # Loss - Fase 1
            axes[0, 0].plot(epochs1, history_phase1.history['loss'], 'b-', label='Train Loss (Phase 1)')
            axes[0, 0].plot(epochs1, history_phase1.history['val_loss'], 'b--', label='Val Loss (Phase 1)')
            
            # Métrica - Fase 1
            axes[1, 0].plot(epochs1, history_phase1.history[metric_name], 'b-', label=f'Train {metric_name} (Phase 1)')
            axes[1, 0].plot(epochs1, history_phase1.history[f'val_{metric_name}'], 'b--', label=f'Val {metric_name} (Phase 1)')
        
        # Fase 2
        if history_phase2:
            offset = len(history_phase1.history['loss']) if history_phase1 else 0
            epochs2 = range(offset + 1, offset + len(history_phase2.history['loss']) + 1)
            
            # Loss - Fase 2
            axes[0, 0].plot(epochs2, history_phase2.history['loss'], 'r-', label='Train Loss (Phase 2 - Fine-tuning)')
            axes[0, 0].plot(epochs2, history_phase2.history['val_loss'], 'r--', label='Val Loss (Phase 2 - Fine-tuning)')
            
            # Métrica - Fase 2
            axes[1, 0].plot(epochs2, history_phase2.history[metric_name], 'r-', label=f'Train {metric_name} (Phase 2)')
            axes[1, 0].plot(epochs2, history_phase2.history[f'val_{metric_name}'], 'r--', label=f'Val {metric_name} (Phase 2)')
        
        axes[0, 0].set_title('Loss durante entrenamiento')
        axes[0, 0].set_xlabel('Época')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        axes[1, 0].set_title(f'{metric_name.upper()} durante entrenamiento')
        axes[1, 0].set_xlabel('Época')
        axes[1, 0].set_ylabel(metric_name.upper())
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Learning rate (si está disponible)
        if history_phase2 and 'lr' in history_phase2.history:
            offset = len(history_phase1.history['loss']) if history_phase1 else 0
            epochs_lr = range(offset + 1, offset + len(history_phase2.history['lr']) + 1)
            axes[0, 1].plot(epochs_lr, history_phase2.history['lr'], 'g-')
            axes[0, 1].set_title('Learning Rate')
            axes[0, 1].set_xlabel('Época')
            axes[0, 1].set_ylabel('LR')
            axes[0, 1].set_yscale('log')
            axes[0, 1].grid(True)
        
        # Resumen de fases
        summary_text = f"Transfer Learning Strategy\n\n"
        summary_text += f"Model: {self.base_model_name}\n"
        summary_text += f"Task: {self.task}\n"
        summary_text += f"Views per molecule: {self.num_views}\n\n"
        
        if history_phase1:
            summary_text += f"Phase 1 (Frozen Base):\n"
            summary_text += f"  Epochs: {len(history_phase1.history['loss'])}\n"
            summary_text += f"  Best val_loss: {min(history_phase1.history['val_loss']):.4f}\n\n"
        
        if history_phase2:
            summary_text += f"Phase 2 (Fine-tuning):\n"
            summary_text += f"  Epochs: {len(history_phase2.history['loss'])}\n"
            summary_text += f"  Best val_loss: {min(history_phase2.history['val_loss']):.4f}\n"
        
        axes[1, 1].text(0.1, 0.5, summary_text, fontsize=11, verticalalignment='center', 
                       family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Gráficas guardadas en: {save_path}")
        plt.close()
    
    def save_model(self, model, filepath='qsar_model_final.keras'):
        """Guardar el modelo completo"""
        model.save(filepath)
        print(f"✓ Modelo guardado en: {filepath}")
    
    def load_model(self, filepath):
        """Cargar un modelo guardado"""
        self.model = keras.models.load_model(filepath)
        print(f"✓ Modelo cargado desde: {filepath}")
        return self.model


class MolecularDataLoader:
    """
    Cargador de datos para imágenes moleculares
    Maneja múltiples vistas por molécula
    """
    
    def __init__(self, 
                 data_dir,
                 labels_file=None,
                 img_size=(224, 224),
                 num_views=6):
        """
        Args:
            data_dir: directorio con estructura: data_dir/molecule_name/1D.png, 2D.png, ..., 6D.png
            labels_file: archivo CSV con columnas: molecule_name, activity (IC50, pIC50, o clase)
            img_size: tamaño de redimensión de imágenes
            num_views: número de vistas por molécula
        """
        self.data_dir = Path(data_dir)
        self.labels_file = labels_file
        self.img_size = img_size
        self.num_views = num_views
        self.molecules = []
        self.labels = {}
        
    def load_labels(self):
        """Cargar etiquetas desde archivo CSV"""
        if self.labels_file and os.path.exists(self.labels_file):
            df = pd.read_csv(self.labels_file)
            self.labels = dict(zip(df['molecule_name'], df['activity']))
            print(f"✓ Etiquetas cargadas: {len(self.labels)} moléculas")
        else:
            print("⚠ Archivo de etiquetas no encontrado, usando valores dummy")
    
    def scan_molecules(self):
        """Escanear directorio para encontrar moléculas con todas sus vistas"""
        
        molecule_dirs = [d for d in self.data_dir.iterdir() if d.is_dir()]
        
        for mol_dir in molecule_dirs:
            # Verificar que tenga todas las vistas (1D.png hasta 6D.png)
            views = []
            for i in range(1, self.num_views + 1):
                view_path = mol_dir / f"{i}D.png"
                if view_path.exists():
                    views.append(str(view_path))
                else:
                    break
            
            if len(views) == self.num_views:
                mol_name = mol_dir.name
                self.molecules.append({
                    'name': mol_name,
                    'views': views,
                    'label': self.labels.get(mol_name, 0.0)  # Default 0.0 si no hay etiqueta
                })
        
        print(f"✓ Moléculas encontradas: {len(self.molecules)}")
        return self.molecules
    
    def load_single_molecule(self, molecule_data):
        """
        Cargar todas las vistas de una molécula
        
        Returns:
            tuple: (list of images, label)
        """
        images = []
        for view_path in molecule_data['views']:
            img = load_img(view_path, target_size=self.img_size)
            img_array = img_to_array(img)
            img_array = img_array / 255.0  # Normalización
            images.append(img_array)
        
        return images, molecule_data['label']
    
    def create_dataset(self, molecules, batch_size=32):
        """
        Crear dataset de TensorFlow para entrenamiento
        
        Returns:
            tf.data.Dataset
        """
        
        def generator():
            for mol_data in molecules:
                views, label = self.load_single_molecule(mol_data)
                # Para multi-view, devolver tupla de vistas
                yield tuple(views), label
        
        # Definir las formas de salida
        output_signature = (
            tuple([tf.TensorSpec(shape=(*self.img_size, 3), dtype=tf.float32) 
                   for _ in range(self.num_views)]),
            tf.TensorSpec(shape=(), dtype=tf.float32)
        )
        
        dataset = tf.data.Dataset.from_generator(
            generator,
            output_signature=output_signature
        )
        
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    def train_val_test_split(self, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_seed=42):
        """
        Dividir datos en conjuntos de entrenamiento, validación y prueba
        
        Returns:
            tuple: (train_molecules, val_molecules, test_molecules)
        """
        
        if not self.molecules:
            self.scan_molecules()
        
        np.random.seed(random_seed)
        indices = np.random.permutation(len(self.molecules))
        
        n_train = int(len(self.molecules) * train_ratio)
        n_val = int(len(self.molecules) * val_ratio)
        
        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:]
        
        train_molecules = [self.molecules[i] for i in train_idx]
        val_molecules = [self.molecules[i] for i in val_idx]
        test_molecules = [self.molecules[i] for i in test_idx]
        
        print(f"✓ Split completado:")
        print(f"  Train: {len(train_molecules)} moléculas")
        print(f"  Validation: {len(val_molecules)} moléculas")
        print(f"  Test: {len(test_molecules)} moléculas")
        
        return train_molecules, val_molecules, test_molecules


def create_dummy_data(output_dir='molecular_data', num_molecules=50):
    """
    Crear datos de ejemplo para demostración
    Genera imágenes sintéticas y etiquetas aleatorias
    """
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print(f"Creando {num_molecules} moléculas de ejemplo...")
    
    labels_data = []
    
    for i in range(num_molecules):
        mol_name = f"MOLECULE_{i+1:04d}"
        mol_dir = output_path / mol_name
        mol_dir.mkdir(exist_ok=True)
        
        # Crear 6 vistas con patrones diferentes
        for view_idx in range(1, 7):
            # Crear imagen sintética (simulando estructura molecular)
            img = np.random.rand(224, 224, 3) * 255
            
            # Agregar formas para simular átomos y enlaces
            center_y, center_x = 112, 112
            for _ in range(10):
                y = np.random.randint(50, 174)
                x = np.random.randint(50, 174)
                color = np.random.rand(3) * 255
                cv2_available = False
                try:
                    import cv2
                    cv2.circle(img, (x, y), 8, color.tolist(), -1)
                    cv2_available = True
                except:
                    pass
                
                if not cv2_available:
                    # Método alternativo sin cv2
                    y_coords, x_coords = np.ogrid[:224, :224]
                    mask = (x_coords - x)**2 + (y_coords - y)**2 <= 8**2
                    img[mask] = color.reshape(1, 3)
            
            # Guardar imagen
            from PIL import Image
            img_pil = Image.fromarray(img.astype('uint8'))
            img_pil.save(mol_dir / f"{view_idx}D.png")
        
        # Generar etiqueta aleatoria (pIC50 entre 4 y 9)
        activity = np.random.uniform(4.0, 9.0)
        labels_data.append({'molecule_name': mol_name, 'activity': activity})
    
    # Guardar labels
    df_labels = pd.DataFrame(labels_data)
    labels_path = output_path / 'labels.csv'
    df_labels.to_csv(labels_path, index=False)
    
    print(f"✓ Datos de ejemplo creados en: {output_path}")
    print(f"✓ Etiquetas guardadas en: {labels_path}")
    
    return str(output_path), str(labels_path)


def main_training_pipeline(data_dir, 
                           labels_file,
                           base_model='VGG16',
                           num_views=6,
                           task='regression'):
    """
    Pipeline completo de entrenamiento con transfer learning en 2 fases
    
    Args:
        data_dir: directorio con imágenes moleculares
        labels_file: archivo CSV con etiquetas
        base_model: 'VGG16' o 'ResNet50'
        num_views: número de vistas por molécula
        task: 'regression' o 'classification'
    """
    
    print("\n" + "="*70)
    print(f"PIPELINE DE TRANSFER LEARNING PARA QSAR - Modelo: {base_model}")
    print("="*70 + "\n")
    
    # 1. Cargar datos
    print("PASO 1: Cargando datos moleculares...")
    data_loader = MolecularDataLoader(
        data_dir=data_dir,
        labels_file=labels_file,
        img_size=(224, 224),
        num_views=num_views
    )
    
    data_loader.load_labels()
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
    batch_size = 8  # Ajustar según memoria disponible
    
    train_dataset = data_loader.create_dataset(train_mols, batch_size=batch_size)
    val_dataset = data_loader.create_dataset(val_mols, batch_size=batch_size)
    test_dataset = data_loader.create_dataset(test_mols, batch_size=batch_size)
    
    # 4. Crear modelo
    print("\nPASO 4: Construyendo modelo...")
    qsar_model = MolecularQSARModel(
        base_model_name=base_model,
        input_shape=(224, 224, 3),
        num_views=num_views,
        task=task,
        num_classes=1
    )
    
    # Construir arquitectura multi-view
    model = qsar_model.build_multi_view_model(freeze_base=True)
    model.summary()
    
    # 5. FASE 1: Entrenar con base congelada
    print("\n" + "="*70)
    print("INICIANDO FASE 1: Capas Convolucionales CONGELADAS")
    print("="*70)
    
    history_phase1 = qsar_model.train_phase1_frozen_base(
        model=model,
        train_data=train_dataset,
        val_data=val_dataset,
        epochs=15,
        batch_size=batch_size,
        learning_rate=1e-3
    )
    
    # Guardar modelo después de fase 1
    qsar_model.save_model(model, 'qsar_model_phase1.keras')
    
    # 6. FASE 2: Fine-tuning
    print("\n" + "="*70)
    print("INICIANDO FASE 2: FINE-TUNING (descongelando capas)")
    print("="*70)
    
    history_phase2 = qsar_model.train_phase2_fine_tuning(
        model=model,
        train_data=train_dataset,
        val_data=val_dataset,
        epochs=25,
        batch_size=batch_size,
        learning_rate=1e-5,
        unfreeze_from_layer=None  # Descongelar todas las capas
    )
    
    # 7. Guardar modelo final
    print("\nPASO 7: Guardando modelo final...")
    qsar_model.save_model(model, 'qsar_model_final.keras')
    
    # 8. Visualizar resultados
    print("\nPASO 8: Generando visualizaciones...")
    qsar_model.plot_training_history(history_phase1, history_phase2, 
                                    save_path='training_history.png')
    
    # 9. Evaluar en test set
    print("\nPASO 9: Evaluación en conjunto de prueba...")
    test_results = model.evaluate(test_dataset, verbose=1)
    
    print("\n" + "="*70)
    print("RESULTADOS FINALES EN TEST SET:")
    print("="*70)
    for metric_name, metric_value in zip(model.metrics_names, test_results):
        print(f"  {metric_name}: {metric_value:.4f}")
    
    print("\n✓ Pipeline completado exitosamente!")
    
    return model, history_phase1, history_phase2, test_results


if __name__ == "__main__":
    
    # Ejemplo de uso
    print("Sistema QSAR con Transfer Learning para predicción de actividad antiinflamatoria")
    print("Inhibición de 5-Lipoxigenasa usando múltiples vistas 3D de moléculas\n")
    
    # Opción 1: Usar datos reales
    # data_dir = 'path/to/your/molecular_data'
    # labels_file = 'path/to/your/labels.csv'
    
    # Opción 2: Crear datos de ejemplo para demostración
    print("Creando datos de ejemplo para demostración...")
    data_dir, labels_file = create_dummy_data(output_dir='molecular_data_demo', num_molecules=30)
    
    # Ejecutar pipeline de entrenamiento
    print("\nEjecutando pipeline de transfer learning...\n")
    
    # Puedes elegir 'VGG16' o 'ResNet50'
    model, hist1, hist2, test_results = main_training_pipeline(
        data_dir=data_dir,
        labels_file=labels_file,
        base_model='VGG16',  # Cambiar a 'ResNet50' si prefieres
        num_views=6,
        task='regression'
    )
    
    print("\n" + "="*70)
    print("INFORMACIÓN SOBRE EL MODELO:")
    print("="*70)
    print("""
    Este sistema implementa un modelo QSAR usando Deep Learning con las siguientes características:
    
    1. ARQUITECTURA:
       - Modelo base pre-entrenado (VGG16 o ResNet50 en ImageNet)
       - Procesamiento de 6 vistas 3D por molécula
       - Fusión de características multi-vista
       - Capas fully connected personalizadas
    
    2. ESTRATEGIA DE TRANSFER LEARNING (2 FASES):
       
       FASE 1 - Capas Convolucionales CONGELADAS:
       - Se congelan todas las capas del modelo base
       - Solo se entrenan las nuevas capas fully connected
       - Learning rate alto (1e-3)
       - Permite adaptar el modelo a las características moleculares
       
       FASE 2 - FINE-TUNING:
       - Se descongelan todas las capas
       - Se re-entrena el modelo completo
       - Learning rate muy bajo (1e-5)
       - Ajuste fino para optimizar predicción de actividad
    
    3. APLICACIÓN:
       - Predicción de actividad antiinflamatoria (inhibición 5-LOX)
       - Entrada: 6 vistas 3D de estructura molecular
       - Salida: pIC50 o valor de actividad (regresión)
       - Alternativa: Clasificación activo/inactivo
    
    4. VENTAJAS:
       - Aprovecha conocimiento pre-entrenado en ImageNet
       - Reduce necesidad de grandes datasets moleculares
       - Captura características visuales de estructuras 3D
       - Estrategia de 2 fases previene overfitting
    """)
    
    print("\nArchivos generados:")
    print("  - qsar_model_phase1.keras: Modelo después de Fase 1")
    print("  - qsar_model_final.keras: Modelo final después de Fase 2")
    print("  - best_model_phase1.keras: Mejor modelo de Fase 1")
    print("  - best_model_phase2.keras: Mejor modelo de Fase 2")
    print("  - training_history.png: Gráficas de entrenamiento")
    print("  - molecular_data_demo/: Datos de ejemplo generados")

import numpy as np
import os
import glob

def normalize_min_max(volume):
    """
    Aplica a normalização Min-Max dinâmica para reescalar os dados para [0, 1].
    """
    vol_min = np.min(volume)
    vol_max = np.max(volume)
    
    if vol_max - vol_min == 0:
        return volume - vol_min
        
    return (volume - vol_min) / (vol_max - vol_min)

def process_dataset(source_folder, dest_folder):
    """
    Varre a pasta em busca de todos os volumes do dataset, fatia e normaliza.
    """
    os.makedirs(dest_folder, exist_ok=True)

    search_pattern = os.path.join(source_folder, "*_fdk_low_dose_256.npy")
    input_files = sorted(glob.glob(search_pattern))

    if not input_files:
        print(f"Erro: Nenhum arquivo '_fdk_low_dose_256.npy' encontrado em {source_folder}")
        return

    print(f"Iniciando processamento em lote. Volumes encontrados: {len(input_files)}\n")

    for input_path in input_files:
        
        filename = os.path.basename(input_path)
    
        prefix = filename.split('_')[0]
        
        paciente_id = f"paciente{prefix}"
        
        target_filename = f"{prefix}_fdk_clinical_dose_256.npy"
        target_path = os.path.join(source_folder, target_filename)

        if not os.path.exists(target_path):
            print(f"-> Aviso: Alvo '{target_filename}' não encontrado. Pulando paciente {prefix}...")
            continue

        print(f"Processando {paciente_id}...")
        
        vol_input = np.load(input_path)
        vol_target = np.load(target_path)

        vol_input_norm = normalize_min_max(vol_input)
        vol_target_norm = normalize_min_max(vol_target)

        num_slices = vol_input_norm.shape[0]

        for slice_idx in range(num_slices):
            slice_input = vol_input_norm[slice_idx, :, :]
            slice_target = vol_target_norm[slice_idx, :, :]
            
            input_savename = f"{paciente_id}_{slice_idx}_input.npy"
            target_savename = f"{paciente_id}_{slice_idx}_target.npy"
            
            np.save(os.path.join(dest_folder, input_savename), slice_input)
            np.save(os.path.join(dest_folder, target_savename), slice_target)
            
    print("\nProcessamento em lote concluído com sucesso!")
    print(f"Todas as fatias 2D estão salvas em: {dest_folder}")

if __name__ == "__main__":
    PASTA_ORIGEM = "../../dataset/new_images" 
    PASTA_DESTINO = "../../dataset/new_images_split"
    
    process_dataset(PASTA_ORIGEM, PASTA_DESTINO)
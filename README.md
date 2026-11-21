# Servidor distribuido de pipelines de entrenamiento

## Fase 1

El coordinador en Go persiste la cola en JSON y asigna trabajos mediante gRPC. Los workers en Python solicitan trabajos según los cupos disponibles y ejecutan Ridge/Lasso, MLP y CNN en CPU. Los checkpoints de MLP y CNN se guardan por época en un volumen compartido. El coordinador reasigna los trabajos cuando un worker pierde el heartbeat durante 6 s o cuando vence el plazo del modelo (30 s, 180 s y 600 s, respectivamente).

### Prueba en PowerShell

Ejecuta estos comandos desde la raíz con Docker Desktop iniciado.

```powershell
docker compose build
docker run --rm -v "${PWD}:/repo" training-worker:phase1 python /repo/scripts/prepare_data.py --mock --output /repo/data/processed
docker run --rm -e PYTHONPATH=/repo/worker -v "${PWD}:/repo:ro" training-worker:phase1 python /repo/scripts/smoke.py local --data /repo/data/processed
docker compose up -d --scale worker=2 --wait
docker compose ps
Get-Content -Raw scripts/smoke.py | docker compose exec -T worker python - distributed
```

La prueba distribuida imprime el estado, el worker, el identificador y las métricas por modelo. La cola conserva los resultados entre ejecuciones. Para encolar tres trabajos nuevos, cambia la semilla.

```powershell
Get-Content -Raw scripts/smoke.py | docker compose exec -T worker python - distributed --seed 43
docker compose logs coordinator worker
docker compose down
```

Los resultados se guardan por intento bajo `/shared/checkpoints/results/<job_id>/<attempt>/`. Los archivos de datos incluyen el CSV de objetos y dos matrices NumPy, una de imágenes y otra con `scene_token`, `split` y `label`. Las particiones siguen una distribución de 6, 2 y 2 escenas.

### Pruebas del coordinador

```powershell
docker run --rm -v "${PWD}:/src" -w /src golang:1.23.3-bookworm go test ./...
```

Los bindings generados de Go se comparten entre el coordinador y el futuro cliente. Los bindings de Python están en `worker/` y `dashboard/`. Para regenerarlos, ejecuta `sh scripts/generate.sh` con Go 1.23.3 y Python 3.11 tras instalar `scripts/requirements-codegen.txt`.

El modo `--nuscenes-dir` espera una copia local de nuScenes-mini con la estructura original del dataset. Instala `scripts/requirements-nuscenes.txt` en un entorno de Python 3.11 y ejecuta `python scripts/prepare_data.py --nuscenes-dir <ruta>`.

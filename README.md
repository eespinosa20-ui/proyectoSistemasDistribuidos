# Servidor distribuido de pipelines de entrenamiento

El coordinador, los workers, el cliente y el dashboard usan Python 3.11. Los workers ejecutan trabajos completos de Ridge/Lasso, MLP y CNN en CPU. MLP y CNN guardan checkpoints por época en el volumen compartido. El coordinador conserva la cola en `/state/queue.json` y reasigna un trabajo cuando su worker supera 6 s sin heartbeat o cuando vence el plazo del modelo. Los plazos son 30 s para regresión lineal, 180 s para MLP y 600 s para CNN.

## Clúster y prueba distribuida

Desde la raíz del repositorio, con Docker Desktop iniciado, ejecuta en PowerShell:

```powershell
docker compose up -d --build --scale worker=2 --wait
docker compose ps
Get-Content -Raw scripts/smoke.py | docker compose exec -T worker python - distributed
docker compose exec -T coordinator python -m unittest -v
```

El dashboard está en [http://127.0.0.1:8080](http://127.0.0.1:8080). Muestra workers, asignaciones, estados y métricas mediante SSE. La pérdida de entrenamiento de MLP y CNN aparece al terminar cada época. Ridge y Lasso reportan sus métricas al completar el ajuste. El contador `REASSIGNED` acumula reasignaciones y el trabajo vuelve a `PENDING`.

Los resultados quedan en `/shared/checkpoints/results/<job_id>/<attempt>/`. El smoke imprime los tres estados y sus métricas. Cambia `--seed` para entrenar tres trabajos nuevos aunque ya existan resultados con la semilla anterior.

```powershell
Get-Content -Raw scripts/smoke.py | docker compose exec -T worker python - distributed --seed 43
docker compose logs coordinator worker dashboard
docker compose down
```

Si necesitas crear datos de demostración en una carpeta vacía, ejecuta `python scripts/prepare_data.py --mock --output data/processed` con las dependencias del worker. Para usar nuScenes-mini instala también `scripts/requirements-nuscenes.txt` y ejecuta `python scripts/prepare_data.py --nuscenes-dir <ruta>`.

## Cliente Python

Instala las dependencias del cliente en un entorno Python 3.11. El cliente usa `127.0.0.1:50051` por defecto. Puedes indicar otro destino con `--address` antes del subcomando o con la variable `COORDINATOR`.

```powershell
python -m pip install -r client/requirements.txt
python --% client/cli.py submit linear_regression --params "{\"variant\":\"ridge\",\"alpha\":0.1}"
python --% client/cli.py submit mlp --params "{\"hidden_layers\":[16,8]}" --grid "{\"epochs\":[2,5],\"learning_rate\":[0.001,0.01]}"
python client/cli.py status JOB_ID_1 JOB_ID_2
```

## Benchmark de 34 trabajos

El conjunto contiene 8 trabajos Ridge/Lasso, 18 MLP y 8 CNN. Instala las dependencias CPU de los pipelines para que `--baseline` pueda ejecutarlos directamente, en secuencia, en el Python local. Usa la misma versión de datos y semilla en ambos modos.

```powershell
python -m pip install -r worker/requirements.txt
python client/cli.py benchmark --baseline --distributed --seed 73 --data data/processed --output benchmark
```

También puedes ejecutar los modos por separado usando el mismo `--output`, `--seed` y `--data-version`. El CLI guarda los tiempos y métricas en `benchmark.json` y `benchmark.csv`. Calcula la reducción porcentual cuando ambas mediciones son comparables. Para repetir la medición distribuida, usa una semilla nueva en ambos modos. Si se reutilizan resultados o hay trabajos fallidos, se guardan las métricas y los tiempos, y el porcentaje queda vacío.

## Regenerar stubs

Requiere Python 3.11 y un shell POSIX. El script genera los stubs Python desde `proto/training.proto` y los copia a los cuatro componentes.

```sh
python -m pip install -r scripts/requirements-codegen.txt
sh scripts/generate.sh
```

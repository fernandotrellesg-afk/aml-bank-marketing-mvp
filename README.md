# MVP · Registro de clientes — Bank Marketing

Proyecto final de **Applied Machine Learning (UTEC, 2026-2)**.

App en Streamlit para registrar clientes en la base de datos del proyecto (dataset *Bank Marketing*, UCI) y obtener la recomendación del modelo final (Random Forest, AUC-PR 0.494 frente a una tasa base de 0.113): **llamar / no llamar**, con un umbral de 0.296 que maximiza el beneficio.

- `app.py`: la app (registrar cliente · ver y descargar la base · carga masiva CSV · sobre el modelo).
- `train_model.py`: reentrena `modelo/rf_bank.joblib` con la misma preparación del informe.
- Base de datos: SQLite (`datos_mvp/clientes.db`), creada con los 41 188 registros originales la primera vez que se abre la app.

```bash
pip install -r requirements.txt
streamlit run app.py
```

Dataset: Moro, Cortez & Rita (2014), UCI Machine Learning Repository, CC BY 4.0.

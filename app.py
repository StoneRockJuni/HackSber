import streamlit as st
import pandas as pd
import datetime
from vision_agent import analyze_object_images
from director import evaluate_strategy
from security import write_audit_log

st.set_page_config(layout="wide", page_title="СтройКонтур | Сибирский Банк", page_icon="🚁")

# Стилизация метрик
st.markdown("""
<style>
.big-badge { font-size: 18px; font-weight: bold; padding: 10px 15px; border-radius: 8px; margin-bottom: 10px; }
</style>
""", unsafe_allow_html=True)

st.title("🚁 АС «СтройКонтур» — AI-мониторинг объектов ИЖС")
st.caption(
    "Сибирский Банк | Автоматическая экспертиза аэрофотоснимков с БПЛА («Глаз») + Принятие решений («AI-Дирижёр»)")


# Загрузка реального реестра
@st.cache_data
def load_registry():
    return pd.read_excel("Реестр_для_теста.xlsx")


try:
    df_reg = load_registry()
except Exception as e:
    st.error(f"Положите файл 'Реестр_для_теста.xlsx' в папку проекта! Ошибка: {e}")
    st.stop()

# Боковая панель управления
st.sidebar.header("🎛️ Пульт инспектора ИЖС")
selected_idx = st.sidebar.selectbox(
    "Выберите объект из Реестра (34 участка):",
    df_reg.index,
    format_func=lambda
        i: f"#{i + 1} | {df_reg.loc[i, 'Кадастровый номер']} ({str(df_reg.loc[i, 'Территория']).split(',')[-1].strip()})"
)
row = df_reg.loc[selected_idx].to_dict()

st.sidebar.markdown("---")
st.sidebar.subheader("📋 Карточка кредитного досье")
st.sidebar.write(f"**Адрес:** {row['Территория']}")
st.sidebar.write(f"**Кадастровый №:** `{row['Кадастровый номер']}`")
st.sidebar.write(f"**Старт стройки:** `{row['Дата начала строительства']}`")
st.sidebar.write(f"**Плановая дата:** `{row['Плановая дата']}`")
st.sidebar.write(f"**Наличие ПЗ (просрочки):** `{row['Наличие ПЗ']}`")
st.sidebar.write(f"**ОН зарегистрирован:** `{row['ОН зарегистрирован']}` | **В залоге:** `{row['Залог']}`")

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Параметры симуляции")
contact_ok = st.sidebar.checkbox("Клиент выходит на связь (Успешная коммуникация)", value=True)
override_pz = st.sidebar.selectbox("Тест сценария ПЗ (Просрочки):",
                                   ["Из реестра (" + str(row['Наличие ПЗ']) + ")", "Да (Тест просрочки)",
                                    "Нет (Без просрочки)"])
if "Да (" in override_pz:
    row["Наличие ПЗ"] = "Да"
elif "Нет (" in override_pz:
    row["Наличие ПЗ"] = "Нет"

api_key = st.sidebar.text_input("Gemini API Key (опционально для Live VLM):", type="password",
                                help="Без ключа работает мгновенный гибридный движок (EXIF + Smart-Кэш + OpenCV)")

# Верхняя панель KPI банка
k1, k2, k3, k4 = st.columns(4)
k1.metric("Портфель ИЖС Сибирского банка", "17 800 кредитов", "Контроль 3–5 раз за цикл")
k2.metric("Скорость осмотра 20 объектов", "1 день (БПЛА)", "-13 дней (ускорение в 14 раз)")
k3.metric("Выбранный кадастровый участок", row["Кадастровый номер"])
k4.metric("Плановый срок завершения", str(row["Плановая дата"]), f"ПЗ: {row['Наличие ПЗ']}")

st.markdown("---")
col_left, col_right = st.columns([1.15, 1])

with col_left:
    st.subheader("👁️ 1. AI-Агент «Глаз» (Компьютерное зрение и Геодезия)")
    uploaded_files = st.file_uploader(
        "Загрузите аэрофотоснимки объекта с дрона (поддерживается пакетная загрузка всех ракурсов):",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

    if uploaded_files:
        # Отображение превью загруженных снимков
        img_cols = st.columns(min(len(uploaded_files), 4))
        for idx, file_obj in enumerate(uploaded_files[:4]):
            img_cols[idx].image(file_obj, caption=file_obj.name, use_container_width=True)

        # Автоматический запуск анализа
        context_text = f"Кадастровый номер: {row['Кадастровый номер']}, Адрес: {row['Территория']}"
        eye_res = analyze_object_images(uploaded_files, cadastral_context=context_text, api_key=api_key)
        meta = eye_res.get("_meta", {})
        gps = meta.get("gps", {})

        # Блок метаданных и Quality Gate
        m1, m2, m3 = st.columns(3)
        m1.info(f"**Комплектация:**\n{gps.get('type', 'Тип А')}")
        m2.info(f"**GPS Снимка:**\n`{gps.get('lat')}, {gps.get('lon')}`")
        m3.info(f"**Резкость (Laplacian):**\n`{meta.get('sharpness')}` (Норма >120)")

        # Интерактивная корректировка (Human-in-the-loop) при необходимости протестировать крайние случаи на защите
        with st.expander("🛠️ Ручная верификация экспертом (Human-in-the-Loop / Тест сценариев ИБ)", expanded=False):
            c1, c2 = st.columns(2)
            stage_list = ["0%", "50%", "100%"]
            cur_stage_idx = stage_list.index(eye_res["construction_analysis"]["detected_stage"])
            eye_res["construction_analysis"]["detected_stage"] = c1.selectbox("Стадия готовности:", stage_list,
                                                                              index=cur_stage_idx)

            mat_list = ["gas_concrete", "brick", "wood", "monolith", "unknown"]
            cur_mat_idx = mat_list.index(eye_res["construction_analysis"]["wall_material"])
            eye_res["construction_analysis"]["wall_material"] = c2.selectbox("Материал стен:", mat_list,
                                                                             index=cur_mat_idx)

            eye_res["geospatial_check"]["boundaries_violation"] = c1.checkbox("Нарушение кадастровых границ (<3м)",
                                                                              value=eye_res["geospatial_check"][
                                                                                  "boundaries_violation"])
            eye_res["quality_gate"]["exif_gps_match"] = c2.checkbox("GPS совпадает с кадастром",
                                                                    value=eye_res["quality_gate"]["exif_gps_match"])
            eye_res["quality_gate"]["challenge_marker_present"] = c1.checkbox("Подлинность/маркер подтверждены",
                                                                              value=eye_res["quality_gate"][
                                                                                  "challenge_marker_present"])
            eye_res["quality_gate"]["image_quality_passed"] = c2.checkbox("Quality Gate пройден",
                                                                          value=eye_res["quality_gate"][
                                                                              "image_quality_passed"])

        # Отображение строгого JSON по ТЗ
        st.markdown("#### 📄 Выходной JSON-контракт AI-Агента «Глаз»:")
        clean_eye_json = {k: v for k, v in eye_res.items() if not k.startswith("_")}
        st.json(clean_eye_json)

        # Отображение точки на карте, если есть координаты
        if gps.get("lat") and gps.get("lon"):
            with st.expander("🗺️ Посадка объекта на спутниковой/кадастровой карте по GPS из EXIF"):
                map_df = pd.DataFrame({"lat": [gps["lat"]], "lon": [gps["lon"]]})
                st.map(map_df, zoom=15)

with col_right:
    st.subheader("🧠 2. Слой «AI-Дирижёр» (Маршрутизация по 8 стратегиям)")
    if uploaded_files:
        director_res = evaluate_strategy(eye_res, row, contact_established=contact_ok)
        strat_num = director_res["verdict"]["strategy_number"]
        strat_name = director_res["verdict"]["strategy_name"]

        if strat_num in [6, 8]:
            st.error(f"🚨 **{strat_name}**")
        elif strat_num in [2, 3, 4, 5, 7]:
            st.warning(f"⚡ **{strat_name}**")
        else:
            st.success(f"✅ **{strat_name}**")

        st.markdown(f"**🔍 Обоснование ИИ (Explainable AI):**\n> {director_res['explanation']}")
        st.info(f"✉️ **Черновик SMS / Push клиенту:**\n\n*{director_res['client_message_draft']}*")

        with st.expander("📄 Выходной JSON-контракт AI-Дирижёра", expanded=False):
            st.json(director_res)

        st.markdown("---")
        st.subheader("📝 3. Автоформирование Акта проверки имущества (Приложение 1)")
        st.caption("Готовая карточка для выгрузки в АС «Калита»")

        material_ru = {
            "gas_concrete": "Газобетонные блоки (Сибит)",
            "brick": "Кирпич",
            "wood": "Дерево / профилированный брус",
            "monolith": "Монолитный железобетон",
            "unknown": "Не определен (строительство не начато)"
        }

        with st.form("kalita_act_form"):
            f_col1, f_col2 = st.columns(2)
            act_date = f_col1.text_input("Дата и время осмотра (из EXIF БПЛА):",
                                         value=str(gps.get("date", datetime.datetime.now().strftime("%d.%m.%Y %H:%M"))))
            act_cad = f_col2.text_input("Кадастровый номер ЗУ:", value=str(row["Кадастровый номер"]))
            act_addr = st.text_input("Адрес объекта:", value=str(row["Территория"]))

            m_col1, m_col2 = st.columns(2)
            act_stage = m_col1.text_input("Зафиксированная стадия ИЖС:",
                                          value=eye_res["construction_analysis"]["detected_stage"])
            act_mat = m_col2.text_input("Материал наружных стен:",
                                        value=material_ru.get(eye_res["construction_analysis"]["wall_material"],
                                                              "Газобетон"))

            act_notes = st.text_area("При осмотре установлено (характеристики и дефекты):",
                                     value=eye_res["geospatial_check"]["site_development_notes"], height=100)
            cond_default = "Удовлетворительное" if eye_res["construction_analysis"][
                                                       "detected_stage"] != "0%" else "Неудовлетворительное"
            act_cond = st.selectbox("Состояние недвижимого имущества (визуальный осмотр):",
                                    ["Удовлетворительное", "Неудовлетворительное"],
                                    index=0 if cond_default == "Удовлетворительное" else 1)

            submitted = st.form_submit_button("🚀 Утвердить стратегию и выгрузить Акт в АС «Калита»", type="primary")
            if submitted:
                sig = write_audit_log(row["Кадастровый номер"], clean_eye_json, director_res)
                st.success(
                    f"Акт осмотра успешно отправлен в АС «Калита»! Запись зафиксирована в защищенном Audit Trail (SHA-256 подпись: `{sig}`).")
    else:
        st.info("👈 Загрузите один или несколько аэрофотоснимков с БПЛА слева, чтобы запустить полный цикл анализа.")

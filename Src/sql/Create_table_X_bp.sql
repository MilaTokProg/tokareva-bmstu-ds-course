-- Table: public.X_bp

-- DROP TABLE IF EXISTS public."X_bp";

CREATE TABLE IF NOT EXISTS public."X_bp"
(
    indx integer NOT NULL,
    mat_nap double precision,
    plotn double precision,
    modupr double precision,
    kolotb double precision,
    epoks double precision,
    temperature double precision,
    povpl double precision,
    moduprrast double precision,
    procn double precision,
    potrsmoli double precision,
    CONSTRAINT "X_bp_pkey" PRIMARY KEY (indx)
)

TABLESPACE pg_default;

ALTER TABLE IF EXISTS public."X_bp"
    OWNER to postgres;

COMMENT ON COLUMN public."X_bp".indx
    IS 'Индекс';

COMMENT ON COLUMN public."X_bp".mat_nap
    IS 'Соотношение матрица-наполнитель';

COMMENT ON COLUMN public."X_bp".plotn
    IS 'Плотность, кг/м3';

COMMENT ON COLUMN public."X_bp".modupr
    IS 'модуль упругости, ГПа';

COMMENT ON COLUMN public."X_bp".kolotb
    IS 'Количество отвердителя, м.%';

COMMENT ON COLUMN public."X_bp".epoks
    IS 'Содержание эпоксидных групп,%_2';

COMMENT ON COLUMN public."X_bp".temperature
    IS 'Температура вспышки, С_2';

COMMENT ON COLUMN public."X_bp".povpl
    IS 'Поверхностная плотность, г/м2';

COMMENT ON COLUMN public."X_bp".moduprrast
    IS 'Модуль упругости при растяжении, ГПа';

COMMENT ON COLUMN public."X_bp".procn
    IS 'Прочность при растяжении, МПа';

COMMENT ON COLUMN public."X_bp".potrsmoli
    IS 'Потребление смолы, г/м2';
-- Table: public.X_nup

-- DROP TABLE IF EXISTS public."X_nup";

CREATE TABLE IF NOT EXISTS public."X_nup"
(
    "Indx" integer NOT NULL,
    ugnash double precision,
    stepnash double precision,
    plotnash double precision,
    CONSTRAINT "X_nup_pkey" PRIMARY KEY ("Indx")
)

TABLESPACE pg_default;

ALTER TABLE IF EXISTS public."X_nup"
    OWNER to postgres;

COMMENT ON COLUMN public."X_nup"."Indx"
    IS 'Индекс';

COMMENT ON COLUMN public."X_nup".ugnash
    IS 'Угол нашивки, град';

COMMENT ON COLUMN public."X_nup".stepnash
    IS 'Шаг нашивки';

COMMENT ON COLUMN public."X_nup".plotnash
    IS 'Плотность нашивки';
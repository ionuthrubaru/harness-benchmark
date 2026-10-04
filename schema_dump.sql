--
-- PostgreSQL database dump
--

\restrict nfZffwdhva7alERTW75jYNRc1sBW1tvAaBZRugAqnKhNGyFRaym0IBbCg7I7BFI

-- Dumped from database version 16.13 (Debian 16.13-1.pgdg12+1)
-- Dumped by pg_dump version 16.13 (Debian 16.13-1.pgdg12+1)

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

--
-- Name: uuid-ossp; Type: EXTENSION; Schema: -; Owner: -
--

CREATE EXTENSION IF NOT EXISTS "uuid-ossp" WITH SCHEMA public;


--
-- Name: EXTENSION "uuid-ossp"; Type: COMMENT; Schema: -; Owner: 
--

COMMENT ON EXTENSION "uuid-ossp" IS 'generate universally unique identifiers (UUIDs)';


--
-- Name: update_product_search_vector(); Type: FUNCTION; Schema: public; Owner: postgres
--

CREATE FUNCTION public.update_product_search_vector() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
    NEW.search_vector :=
        setweight(to_tsvector('english', COALESCE(NEW.name, '')),        'A') ||
        setweight(to_tsvector('english', COALESCE(NEW.description, '')), 'B') ||
        setweight(to_tsvector('english', COALESCE(NEW.product_type, '')), 'C');
    RETURN NEW;
END;
$$;


ALTER FUNCTION public.update_product_search_vector() OWNER TO postgres;

SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- Name: events; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.events (
    id uuid DEFAULT public.uuid_generate_v4() NOT NULL,
    user_id uuid NOT NULL,
    event_type character varying(50) NOT NULL,
    product_id uuid,
    session_id character varying(64),
    metadata jsonb DEFAULT '{}'::jsonb NOT NULL,
    occurred_at timestamp with time zone DEFAULT now() NOT NULL
);


ALTER TABLE public.events OWNER TO postgres;

--
-- Name: invoice_lines; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.invoice_lines (
    id uuid DEFAULT public.uuid_generate_v4() NOT NULL,
    invoice_id uuid NOT NULL,
    product_id uuid,
    description character varying(500) NOT NULL,
    quantity smallint DEFAULT 1 NOT NULL,
    unit_price_usd numeric(8,2) NOT NULL,
    line_total_usd numeric(10,2) NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL
);


ALTER TABLE public.invoice_lines OWNER TO postgres;

--
-- Name: invoices; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.invoices (
    id uuid DEFAULT public.uuid_generate_v4() NOT NULL,
    user_id uuid NOT NULL,
    invoice_type character varying(20) NOT NULL,
    status character varying(20) DEFAULT 'pending'::character varying NOT NULL,
    subtotal_usd numeric(10,2) NOT NULL,
    tax_usd numeric(10,2) DEFAULT 0.00 NOT NULL,
    discount_usd numeric(10,2) DEFAULT 0.00 NOT NULL,
    total_usd numeric(10,2) NOT NULL,
    subscription_id uuid,
    billing_period_start timestamp with time zone,
    billing_period_end timestamp with time zone,
    paid_at timestamp with time zone,
    due_at timestamp with time zone NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL
);


ALTER TABLE public.invoices OWNER TO postgres;

--
-- Name: order_items; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.order_items (
    id uuid DEFAULT public.uuid_generate_v4() NOT NULL,
    order_id uuid NOT NULL,
    product_id uuid NOT NULL,
    quantity smallint DEFAULT 1 NOT NULL,
    unit_price_usd numeric(8,2) NOT NULL,
    line_total_usd numeric(10,2) NOT NULL,
    fulfilment_status character varying(20) DEFAULT 'pending'::character varying NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL
);


ALTER TABLE public.order_items OWNER TO postgres;

--
-- Name: orders; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.orders (
    id uuid DEFAULT public.uuid_generate_v4() NOT NULL,
    user_id uuid NOT NULL,
    invoice_id uuid NOT NULL,
    status character varying(20) DEFAULT 'pending'::character varying NOT NULL,
    shipping_name character varying(255),
    shipping_address character varying(500),
    shipping_city character varying(100),
    shipping_country character(2),
    shipping_postal character varying(20),
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    updated_at timestamp with time zone DEFAULT now() NOT NULL
);


ALTER TABLE public.orders OWNER TO postgres;

--
-- Name: products; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.products (
    id uuid DEFAULT public.uuid_generate_v4() NOT NULL,
    name character varying(255) NOT NULL,
    slug character varying(255) NOT NULL,
    product_type character varying(20) NOT NULL,
    description text NOT NULL,
    price_usd numeric(8,2) NOT NULL,
    currency character(3) DEFAULT 'USD'::bpchar NOT NULL,
    is_active boolean DEFAULT true NOT NULL,
    seller_id uuid NOT NULL,
    attributes jsonb DEFAULT '{}'::jsonb NOT NULL,
    search_vector tsvector,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    updated_at timestamp with time zone DEFAULT now() NOT NULL
);


ALTER TABLE public.products OWNER TO postgres;

--
-- Name: seller_profiles; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.seller_profiles (
    user_id uuid NOT NULL,
    display_name character varying(255) NOT NULL,
    legal_name character varying(255),
    tax_id character varying(100),
    payout_email character varying(255),
    country_code character(2),
    is_verified boolean DEFAULT false NOT NULL,
    bio text,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    updated_at timestamp with time zone DEFAULT now() NOT NULL
);


ALTER TABLE public.seller_profiles OWNER TO postgres;

--
-- Name: sessions; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.sessions (
    id character varying(64) NOT NULL,
    user_id uuid NOT NULL,
    cart jsonb DEFAULT '[]'::jsonb NOT NULL,
    ip_address inet,
    user_agent character varying(500),
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    last_active_at timestamp with time zone DEFAULT now() NOT NULL,
    expires_at timestamp with time zone NOT NULL
);


ALTER TABLE public.sessions OWNER TO postgres;

--
-- Name: subscription_tier_pricing; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.subscription_tier_pricing (
    tier_id smallint NOT NULL,
    valid_from timestamp with time zone NOT NULL,
    valid_to timestamp with time zone,
    monthly_price_usd numeric(8,2) NOT NULL,
    is_active boolean DEFAULT true NOT NULL
);


ALTER TABLE public.subscription_tier_pricing OWNER TO postgres;

--
-- Name: subscription_tiers; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.subscription_tiers (
    id smallint NOT NULL,
    name character varying(50) NOT NULL,
    description text,
    features jsonb DEFAULT '{}'::jsonb NOT NULL
);


ALTER TABLE public.subscription_tiers OWNER TO postgres;

--
-- Name: subscriptions; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.subscriptions (
    id uuid DEFAULT public.uuid_generate_v4() NOT NULL,
    user_id uuid NOT NULL,
    tier_id smallint NOT NULL,
    status character varying(20) DEFAULT 'active'::character varying NOT NULL,
    started_at timestamp with time zone DEFAULT now() NOT NULL,
    current_period_start timestamp with time zone NOT NULL,
    current_period_end timestamp with time zone NOT NULL,
    cancelled_at timestamp with time zone,
    cancel_reason text,
    billing_cycle character varying(10) DEFAULT 'monthly'::character varying NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    updated_at timestamp with time zone DEFAULT now() NOT NULL
);


ALTER TABLE public.subscriptions OWNER TO postgres;

--
-- Name: users; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.users (
    id uuid DEFAULT public.uuid_generate_v4() NOT NULL,
    email character varying(255) NOT NULL,
    full_name character varying(255) NOT NULL,
    country_code character(2) NOT NULL,
    city character varying(100),
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    last_login_at timestamp with time zone,
    is_active boolean DEFAULT true NOT NULL,
    preferences jsonb DEFAULT '{}'::jsonb NOT NULL
);


ALTER TABLE public.users OWNER TO postgres;

--
-- Name: events events_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.events
    ADD CONSTRAINT events_pkey PRIMARY KEY (id);


--
-- Name: events_q8 events_q8_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.events_q8
    ADD CONSTRAINT events_q8_pkey PRIMARY KEY (id);


--
-- Name: invoice_lines invoice_lines_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.invoice_lines
    ADD CONSTRAINT invoice_lines_pkey PRIMARY KEY (id);


--
-- Name: invoices invoices_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.invoices
    ADD CONSTRAINT invoices_pkey PRIMARY KEY (id);


--
-- Name: order_items order_items_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.order_items
    ADD CONSTRAINT order_items_pkey PRIMARY KEY (id);


--
-- Name: orders orders_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.orders
    ADD CONSTRAINT orders_pkey PRIMARY KEY (id);


--
-- Name: products products_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.products
    ADD CONSTRAINT products_pkey PRIMARY KEY (id);


--
-- Name: products products_slug_key; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.products
    ADD CONSTRAINT products_slug_key UNIQUE (slug);


--
-- Name: seller_profiles seller_profiles_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.seller_profiles
    ADD CONSTRAINT seller_profiles_pkey PRIMARY KEY (user_id);


--
-- Name: sessions sessions_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.sessions
    ADD CONSTRAINT sessions_pkey PRIMARY KEY (id);


--
-- Name: subscription_tier_pricing subscription_tier_pricing_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.subscription_tier_pricing
    ADD CONSTRAINT subscription_tier_pricing_pkey PRIMARY KEY (tier_id, valid_from);


--
-- Name: subscription_tiers subscription_tiers_name_key; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.subscription_tiers
    ADD CONSTRAINT subscription_tiers_name_key UNIQUE (name);


--
-- Name: subscription_tiers subscription_tiers_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.subscription_tiers
    ADD CONSTRAINT subscription_tiers_pkey PRIMARY KEY (id);


--
-- Name: subscriptions subscriptions_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.subscriptions
    ADD CONSTRAINT subscriptions_pkey PRIMARY KEY (id);


--
-- Name: users users_email_key; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.users
    ADD CONSTRAINT users_email_key UNIQUE (email);


--
-- Name: users users_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.users
    ADD CONSTRAINT users_pkey PRIMARY KEY (id);


--
-- Name: idx_events_occurred_at; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_events_occurred_at ON public.events USING btree (occurred_at DESC);


--
-- Name: idx_events_product_id; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_events_product_id ON public.events USING btree (product_id);


--
-- Name: idx_events_type; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_events_type ON public.events USING btree (event_type);


--
-- Name: idx_events_user_time; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_events_user_time ON public.events USING btree (user_id, occurred_at DESC);


--
-- Name: idx_invoice_lines_invoice_id; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_invoice_lines_invoice_id ON public.invoice_lines USING btree (invoice_id);


--
-- Name: idx_invoice_lines_product_id; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_invoice_lines_product_id ON public.invoice_lines USING btree (product_id);


--
-- Name: idx_invoices_created_at; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_invoices_created_at ON public.invoices USING btree (created_at);


--
-- Name: idx_invoices_status; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_invoices_status ON public.invoices USING btree (status);


--
-- Name: idx_invoices_subscription_id; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_invoices_subscription_id ON public.invoices USING btree (subscription_id);


--
-- Name: idx_invoices_type; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_invoices_type ON public.invoices USING btree (invoice_type);


--
-- Name: idx_invoices_type_status_created; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_invoices_type_status_created ON public.invoices USING btree (invoice_type, status, created_at);


--
-- Name: idx_invoices_user_id; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_invoices_user_id ON public.invoices USING btree (user_id);


--
-- Name: idx_order_items_order_id; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_order_items_order_id ON public.order_items USING btree (order_id);


--
-- Name: idx_order_items_product_id; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_order_items_product_id ON public.order_items USING btree (product_id);


--
-- Name: idx_orders_created_at; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_orders_created_at ON public.orders USING btree (created_at);


--
-- Name: idx_orders_invoice_id; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_orders_invoice_id ON public.orders USING btree (invoice_id);


--
-- Name: idx_orders_status; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_orders_status ON public.orders USING btree (status);


--
-- Name: idx_orders_user_id; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_orders_user_id ON public.orders USING btree (user_id);


--
-- Name: idx_products_active; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_products_active ON public.products USING btree (is_active);


--
-- Name: idx_products_attributes; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_products_attributes ON public.products USING gin (attributes);


--
-- Name: idx_products_price; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_products_price ON public.products USING btree (price_usd);


--
-- Name: idx_products_search; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_products_search ON public.products USING gin (search_vector);


--
-- Name: idx_products_seller_id; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_products_seller_id ON public.products USING btree (seller_id);


--
-- Name: idx_products_type; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_products_type ON public.products USING btree (product_type);


--
-- Name: idx_seller_profiles_country; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_seller_profiles_country ON public.seller_profiles USING btree (country_code);


--
-- Name: idx_seller_profiles_verified; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_seller_profiles_verified ON public.seller_profiles USING btree (is_verified);


--
-- Name: idx_sessions_expires_at; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_sessions_expires_at ON public.sessions USING btree (expires_at);


--
-- Name: idx_sessions_last_active; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_sessions_last_active ON public.sessions USING btree (last_active_at);


--
-- Name: idx_sessions_user_id; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_sessions_user_id ON public.sessions USING btree (user_id);


--
-- Name: idx_subscriptions_one_active_per_user; Type: INDEX; Schema: public; Owner: postgres
--

CREATE UNIQUE INDEX idx_subscriptions_one_active_per_user ON public.subscriptions USING btree (user_id) WHERE ((status)::text = 'active'::text);


--
-- Name: idx_subscriptions_period_end; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_subscriptions_period_end ON public.subscriptions USING btree (current_period_end);


--
-- Name: idx_subscriptions_status; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_subscriptions_status ON public.subscriptions USING btree (status);


--
-- Name: idx_subscriptions_tier_id; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_subscriptions_tier_id ON public.subscriptions USING btree (tier_id);


--
-- Name: idx_subscriptions_user_id; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_subscriptions_user_id ON public.subscriptions USING btree (user_id);


--
-- Name: idx_tier_pricing_one_active_per_tier; Type: INDEX; Schema: public; Owner: postgres
--

CREATE UNIQUE INDEX idx_tier_pricing_one_active_per_tier ON public.subscription_tier_pricing USING btree (tier_id) WHERE (is_active = true);


--
-- Name: idx_tier_pricing_tier_id; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_tier_pricing_tier_id ON public.subscription_tier_pricing USING btree (tier_id);


--
-- Name: idx_tier_pricing_valid_from; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_tier_pricing_valid_from ON public.subscription_tier_pricing USING btree (valid_from);


--
-- Name: idx_users_country; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_users_country ON public.users USING btree (country_code);


--
-- Name: idx_users_created_at; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_users_created_at ON public.users USING btree (created_at);


--
-- Name: idx_users_email; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_users_email ON public.users USING btree (email);


--
-- Name: products trg_products_search_vector; Type: TRIGGER; Schema: public; Owner: postgres
--

CREATE TRIGGER trg_products_search_vector BEFORE INSERT OR UPDATE ON public.products FOR EACH ROW EXECUTE FUNCTION public.update_product_search_vector();


--
-- Name: events events_product_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.events
    ADD CONSTRAINT events_product_id_fkey FOREIGN KEY (product_id) REFERENCES public.products(id);


--
-- Name: events events_session_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.events
    ADD CONSTRAINT events_session_id_fkey FOREIGN KEY (session_id) REFERENCES public.sessions(id);


--
-- Name: events events_user_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.events
    ADD CONSTRAINT events_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.users(id);


--
-- Name: invoice_lines invoice_lines_invoice_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.invoice_lines
    ADD CONSTRAINT invoice_lines_invoice_id_fkey FOREIGN KEY (invoice_id) REFERENCES public.invoices(id) ON DELETE CASCADE;


--
-- Name: invoice_lines invoice_lines_product_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.invoice_lines
    ADD CONSTRAINT invoice_lines_product_id_fkey FOREIGN KEY (product_id) REFERENCES public.products(id);


--
-- Name: invoices invoices_subscription_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.invoices
    ADD CONSTRAINT invoices_subscription_id_fkey FOREIGN KEY (subscription_id) REFERENCES public.subscriptions(id);


--
-- Name: invoices invoices_user_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.invoices
    ADD CONSTRAINT invoices_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.users(id);


--
-- Name: order_items order_items_order_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.order_items
    ADD CONSTRAINT order_items_order_id_fkey FOREIGN KEY (order_id) REFERENCES public.orders(id) ON DELETE CASCADE;


--
-- Name: order_items order_items_product_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.order_items
    ADD CONSTRAINT order_items_product_id_fkey FOREIGN KEY (product_id) REFERENCES public.products(id);


--
-- Name: orders orders_invoice_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.orders
    ADD CONSTRAINT orders_invoice_id_fkey FOREIGN KEY (invoice_id) REFERENCES public.invoices(id);


--
-- Name: orders orders_user_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.orders
    ADD CONSTRAINT orders_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.users(id);


--
-- Name: products products_seller_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.products
    ADD CONSTRAINT products_seller_id_fkey FOREIGN KEY (seller_id) REFERENCES public.users(id);


--
-- Name: seller_profiles seller_profiles_user_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.seller_profiles
    ADD CONSTRAINT seller_profiles_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.users(id) ON DELETE CASCADE;


--
-- Name: sessions sessions_user_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.sessions
    ADD CONSTRAINT sessions_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.users(id) ON DELETE CASCADE;


--
-- Name: subscription_tier_pricing subscription_tier_pricing_tier_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.subscription_tier_pricing
    ADD CONSTRAINT subscription_tier_pricing_tier_id_fkey FOREIGN KEY (tier_id) REFERENCES public.subscription_tiers(id);


--
-- Name: subscriptions subscriptions_tier_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.subscriptions
    ADD CONSTRAINT subscriptions_tier_id_fkey FOREIGN KEY (tier_id) REFERENCES public.subscription_tiers(id);


--
-- Name: subscriptions subscriptions_user_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.subscriptions
    ADD CONSTRAINT subscriptions_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.users(id) ON DELETE CASCADE;


--
-- PostgreSQL database dump complete
--

\unrestrict nfZffwdhva7alERTW75jYNRc1sBW1tvAaBZRugAqnKhNGyFRaym0IBbCg7I7BFI


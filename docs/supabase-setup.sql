-- B.S.Y DIGITAL : base de données du site (demandes + projets + images)
-- À coller dans Supabase > SQL Editor > New query, puis "Run".

-- 1. Demandes envoyées par le formulaire de contact
create table if not exists public.leads (
  id uuid primary key default gen_random_uuid(),
  created_at timestamptz not null default now(),
  name text not null check (char_length(name) between 2 and 120),
  phone text not null check (char_length(phone) between 6 and 40),
  email text check (char_length(email) <= 200),
  type text check (char_length(type) <= 80),
  timing text check (char_length(timing) <= 80),
  biz text check (char_length(biz) <= 300),
  site text check (char_length(site) <= 300),
  msg text check (char_length(msg) <= 2000),
  lang text check (char_length(lang) <= 5),
  handled boolean not null default false
);
alter table public.leads enable row level security;
drop policy if exists "visitors send" on public.leads;
drop policy if exists "owner reads" on public.leads;
drop policy if exists "owner updates" on public.leads;
drop policy if exists "owner deletes" on public.leads;
create policy "visitors send" on public.leads for insert to anon, authenticated with check (handled = false);
create policy "owner reads" on public.leads for select to authenticated using ((auth.jwt() ->> 'email') = 'bsydigital@icloud.com');
create policy "owner updates" on public.leads for update to authenticated using ((auth.jwt() ->> 'email') = 'bsydigital@icloud.com') with check ((auth.jwt() ->> 'email') = 'bsydigital@icloud.com');
create policy "owner deletes" on public.leads for delete to authenticated using ((auth.jwt() ->> 'email') = 'bsydigital@icloud.com');

-- 2. Projets ajoutés depuis la page /admin
create table if not exists public.projects (
  id uuid primary key default gen_random_uuid(),
  created_at timestamptz not null default now(),
  name text not null,
  url text not null,
  title text, subtitle text, type text, field text, location text,
  cat text not null default 'other',
  challenge text, built text,
  points jsonb not null default '[]'::jsonb,
  img_d text, img_m text
);
alter table public.projects enable row level security;
drop policy if exists "anyone reads" on public.projects;
drop policy if exists "owner writes" on public.projects;
create policy "anyone reads" on public.projects for select to anon, authenticated using (true);
create policy "owner writes" on public.projects for all to authenticated using ((auth.jwt() ->> 'email') = 'bsydigital@icloud.com') with check ((auth.jwt() ->> 'email') = 'bsydigital@icloud.com');

-- 3. Dossier d'images des projets (lecture publique, envoi réservé au propriétaire)
insert into storage.buckets (id, name, public) values ('projects', 'projects', true) on conflict (id) do update set public = true;
drop policy if exists "owner uploads" on storage.objects;
drop policy if exists "owner changes" on storage.objects;
drop policy if exists "owner removes" on storage.objects;
create policy "owner uploads" on storage.objects for insert to authenticated with check (bucket_id = 'projects' and (auth.jwt() ->> 'email') = 'bsydigital@icloud.com');
create policy "owner changes" on storage.objects for update to authenticated using (bucket_id = 'projects' and (auth.jwt() ->> 'email') = 'bsydigital@icloud.com');
create policy "owner removes" on storage.objects for delete to authenticated using (bucket_id = 'projects' and (auth.jwt() ->> 'email') = 'bsydigital@icloud.com');

-- NOTE: change to your own passwords for production environments
\set pgpass `echo "$POSTGRES_PASSWORD"`

ALTER USER authenticator WITH PASSWORD :'pgpass';
ALTER USER pgbouncer WITH PASSWORD :'pgpass';
ALTER USER supabase_auth_admin WITH PASSWORD :'pgpass';
ALTER USER supabase_functions_admin WITH PASSWORD :'pgpass';
ALTER USER supabase_storage_admin WITH PASSWORD :'pgpass';

CREATE ROLE n8n_user WITH LOGIN PASSWORD 'your_secure_n8n_password';
CREATE DATABASE n8n OWNER n8n_user;

CREATE ROLE webui_user WITH LOGIN PASSWORD 'your_secure_webui_password';
CREATE DATABASE webui OWNER webui_user;

-- CREATE ROLE postgres WITH LOGIN PASSWORD 'your_secure_supabase_password';
-- CREATE DATABASE postgres OWNER postgres;
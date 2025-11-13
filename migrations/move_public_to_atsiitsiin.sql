-- Migration: move Atsiitsʼiin tables from MS3DM.PUBLIC to MS3DM.ATSIITSIIN
-- Run this script with a role that can manage objects in both schemas.

USE DATABASE MS3DM;

CREATE SCHEMA IF NOT EXISTS ATSIITSIIN;

-- Helper procedure: move table if it exists in PUBLIC
CREATE OR REPLACE PROCEDURE ATSIITSIIN.MOVE_TABLE_IF_EXISTS(table_name STRING)
RETURNS STRING
LANGUAGE JAVASCRIPT
AS
$$
var tableName = TABLE_NAME.trim().toUpperCase();
var fromSchema = 'PUBLIC';
var toSchema = 'ATSIITSIIN';

var sqlCheck = `SHOW TABLES LIKE '${tableName}' IN SCHEMA MS3DM.` + fromSchema;
var checkStmt = snowflake.createStatement({sqlText: sqlCheck});
var result = checkStmt.execute();
if (!result.next()) {
  return `Table ${tableName} not found in ${fromSchema}; skipping.`;
}

var renameSql = `ALTER TABLE MS3DM.${fromSchema}.` + tableName +
                ` RENAME TO MS3DM.${toSchema}.` + tableName;
var renameStmt = snowflake.createStatement({sqlText: renameSql});
renameStmt.execute();
return `Table ${tableName} moved to ${toSchema}.`;
$$;

CALL ATSIITSIIN.MOVE_TABLE_IF_EXISTS('NOTES');
CALL ATSIITSIIN.MOVE_TABLE_IF_EXISTS('NOTE_CHUNKS');
CALL ATSIITSIIN.MOVE_TABLE_IF_EXISTS('NOTE_CONTEXT_WORKLOG');
CALL ATSIITSIIN.MOVE_TABLE_IF_EXISTS('NOTE_SENTIMENT_WORKLOG');
CALL ATSIITSIIN.MOVE_TABLE_IF_EXISTS('TAGS');
CALL ATSIITSIIN.MOVE_TABLE_IF_EXISTS('NOTE_TAGS');
CALL ATSIITSIIN.MOVE_TABLE_IF_EXISTS('NOTE_TAG_AUDIT');

DROP PROCEDURE IF EXISTS ATSIITSIIN.MOVE_TABLE_IF_EXISTS(STRING);

-- Reapply grants for the new schema.
GRANT USAGE ON SCHEMA MS3DM.ATSIITSIIN TO ROLE MS3DM_ACCOUNT_ADMIN;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA MS3DM.ATSIITSIIN TO ROLE MS3DM_ACCOUNT_ADMIN;
GRANT ALL PRIVILEGES ON FUTURE TABLES IN SCHEMA MS3DM.ATSIITSIIN TO ROLE MS3DM_ACCOUNT_ADMIN;


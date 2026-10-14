# Notion MCP 설정

이 문서는 Orca에서 Codex를 사용하는 프로젝트의 Notion MCP 설정 및 문제 해결 절차다. 다른 프로젝트를 설정할 때도 먼저 현재 런타임의 도구 목록과 프로젝트 설정을 확인하고, 연결이 빠졌다면 기존 설정을 보존하면서 복구한다.

## 먼저 확인할 것

1. 현재 에이전트 런타임에서 Notion MCP 도구가 실제로 노출되는지 확인한다. 설정 파일이 있어도 이 세션의 도구 목록에 MCP가 없을 수 있다.
2. ChatGPT 앱에 연결된 Notion과 Orca가 실행한 Codex의 Notion MCP는 별도 연결로 취급한다. 한쪽의 OAuth 인증이 다른 런타임에 자동 공유된다고 가정하지 않는다.
3. Orca의 프로젝트 MCP 화면은 `.mcp.json`을 감지하고 표시할 수 있지만, 실제 로드는 각 에이전트가 지원하는 설정 위치와 형식에 달려 있다.

## Orca + Codex 프로젝트 설정

Codex에서 프로젝트별로 쓸 때는 프로젝트 루트의 `.codex/config.toml`에 다음 항목을 추가한다. 프로젝트가 Codex의 신뢰 대상이어야 프로젝트 설정이 로드된다.

```toml
[mcp_servers.notion]
url = "https://mcp.notion.com/mcp"
```

기존 `.codex/config.toml`이 있으면 덮어쓰지 말고 `[mcp_servers.notion]` 테이블만 병합한다. 이미 `notion` 항목이 있으면 URL, 인증, 활성화 설정을 확인하고 중복을 만들지 않는다.

Orca의 프로젝트 MCP 패널에서 서버를 보이게 하거나 `.mcp.json`을 사용하는 다른 MCP 클라이언트와 공유할 필요가 있으면 루트 `.mcp.json`에 아래 형식을 추가할 수 있다. 이 파일만으로 Codex 런타임이 설정을 로드한다고 가정하지 않는다.

```json
{
  "mcpServers": {
    "notion": {
      "url": "https://mcp.notion.com/mcp"
    }
  }
}
```

## 인증 및 확인

1. 해당 프로젝트의 Codex 런타임에서 `codex mcp list`로 `notion` 항목이 있는지 확인한다.
2. 설정이 없으면 파일을 기존 내용에 병합한 뒤 다시 확인한다.
3. OAuth 로그인이 안 되어 있으면 `codex mcp login notion`을 실행한다. 명령이 제공하는 링크에서 사용자가 Notion에 로그인하고 권한을 승인해야 한다. 사용자의 동의 화면을 대신 승인하지 않는다.
4. `Successfully logged in to MCP server 'notion'`은 OAuth 로그인 완료를 확인한다. `codex mcp list`의 `enabled / OAuth`는 서버 설정과 인증 방식을 보여 주므로, 새 Codex 세션에서 도구가 실제로 노출되는지도 따로 확인한다.
5. 연결 후에도 현재 세션의 도구 목록이 그대로라면 Orca에서 Codex 세션을 새로 시작한다. MCP 설정 파일 변경은 이미 실행 중인 세션에 즉시 반영되지 않을 수 있다.

## 경계 및 주의

- Notion의 공식 원격 MCP 주소는 `https://mcp.notion.com/mcp`이며 OAuth를 사용한다. ChatGPT에서 이미 연결했다는 사실만으로 Orca/Codex 인증이 끝났다고 보지 않는다.
- 다른 에이전트를 쓰는 프로젝트라면 해당 에이전트의 공식 설정 파일과 형식을 사용한다. Codex의 TOML 형식을 모든 클라이언트에 그대로 적용하지 않는다.
- 기존 MCP 설정을 덮어쓰거나, 인증 토큰을 설정 문서/저장소에 기록하지 않는다.
- 연결 여부를 확인하는 데 필요한 범위에서만 명령을 실행한다. MCP 서버의 쓰기 도구를 호출하는 것은 사용자가 요청한 작업이 있을 때만 한다.

## 이 절차를 확인한 사례

이 프로젝트에서는 빈 `.mcp.json`을 유지한 채 `.codex/config.toml`에 Codex용 설정을 추가하고, 루트 `.mcp.json`에도 Orca 프로젝트 MCP 화면용 서버 항목을 병합했다. `codex mcp list`에서 `notion`이 `enabled / OAuth`로 확인됐고, `codex mcp login notion`은 성공적으로 완료됐다. 새 Codex 세션을 열어 도구 노출을 확인해야 한다.

## 참고 자료

- [Orca MCP 설정 개요](https://www.onorca.dev/docs/cli/skills)
- [Orca의 프로젝트별 MCP 설정 동작](https://github.com/stablyai/orca/issues/10132)
- [Codex 설정 및 MCP](https://developers.openai.com/learn/docs-mcp)
- [Notion MCP 연결 안내](https://www.notion.com/help/notion-mcp)
- [Notion 원격 MCP 주소와 OAuth 요구사항](https://github.com/makenotion/notion-cookbook/blob/main/docs/mcp-client-integration.md)

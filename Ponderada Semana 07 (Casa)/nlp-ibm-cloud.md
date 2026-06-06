# Levantamento de Ferramentas de NLP — IBM Cloud

**Atividade:** levantamento de serviços de Processamento de Linguagem Natural (NLP) disponíveis em uma nuvem comercial. 

**Nuvem escolhida:** IBM Cloud

---

## 1. Visão Geral

A IBM disponibiliza um portfólio de serviços de NLP organizado sob a marca **Watson** e, mais recentemente, sob a plataforma **watsonx**. Os serviços expõem APIs REST autenticadas via IAM (API Key + Bearer token) ou via autenticação básica (`apikey:<APIKEY>`), e retornam respostas em JSON.

A tabela a seguir resume os serviços de NLP catalogados:

| # | Serviço | Categoria | Finalidade principal |
|---|---------|-----------|----------------------|
| 1 | Watson Natural Language Understanding (NLU) | Análise de texto | Extrair metadados (entidades, sentimento, emoção, conceitos, etc.) |
| 2 | watsonx Assistant | Conversacional | Construir assistentes virtuais / chatbots |
| 3 | Watson Discovery | Busca cognitiva | Busca semântica e enriquecimento NLP sobre documentos |
| 4 | Watson Speech to Text | Fala → texto | Transcrição de áudio |
| 5 | Watson Text to Speech | Texto → fala | Síntese de voz |
| 6 | Watson Language Translator | Tradução | Tradução automática entre idiomas |
| 7 | watsonx.ai (Foundation Models) | LLMs | Geração, sumarização, classificação via modelos de fundação |

> **Observação sobre disponibilidade:** Alguns serviços históricos do Watson (como Tone Analyzer e Personality Insights) foram descontinuados pela IBM. O Language Translator continua referenciado na documentação oficial;

---

## 2. Pré-requisitos comuns

Para reproduzir os exemplos:

1. Conta na [IBM Cloud](https://cloud.ibm.com).
2. Instância do serviço criada via [Catálogo](https://cloud.ibm.com/catalog).
3. Coletar `apikey` e `url` da instância em **Manage → Credentials**.
4. Exportar variáveis no terminal:
   ```bash
   export APIKEY="<sua-api-key>"
   export URL="<endpoint-da-instancia>"
   ```

Autenticação básica usada na maioria dos serviços Watson clássicos:
```
-u "apikey:$APIKEY"
```

---

## 3. Serviços e Exemplos de Aplicação

### 3.1 Watson Natural Language Understanding (NLU)

**Descrição.** Serviço para análise de texto, HTML ou URLs públicas, extraindo metadados como categorias, conceitos, emoções, entidades, palavras-chave, relações, papéis semânticos, sentimento e sintaxe.

**Endpoint:**
```
POST {URL}/v1/analyze?version=2022-04-07
```

**Exemplo — análise de sentimento e palavras-chave (cURL):**
```bash
curl -X POST \
  -u "apikey:$APIKEY" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "A nova versão do aplicativo está rápida e fácil de usar.",
    "features": {
      "sentiment": {},
      "keywords": { "limit": 5 }
    },
    "language": "pt"
  }' \
  "$URL/v1/analyze?version=2022-04-07"
```

**Aplicação prática.** Análise de sentimento de comentários de clientes em redes sociais para detectar picos de insatisfação com uma campanha de marca.

---

### 3.2 watsonx Assistant

**Descrição.** Plataforma de construção de assistentes virtuais que combina aprendizado de máquina, NLU e um editor de diálogos para criar fluxos conversacionais entre aplicações e usuários.

**Endpoint (envio de mensagem para sessão existente):**
```
POST {URL}/v2/assistants/{assistant_id}/sessions/{session_id}/message?version=2023-06-15
```

**Exemplo (cURL):**
```bash
curl -X POST \
  -u "apikey:$APIKEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "message_type": "text",
      "text": "Quero saber o horário de funcionamento."
    }
  }' \
  "$URL/v2/assistants/$ASSISTANT_ID/sessions/$SESSION_ID/message?version=2023-06-15"
```

**Aplicação prática.** Chatbot de atendimento ao cliente (SAC) capaz de responder a perguntas frequentes sobre pedidos, horários e políticas de uma rede de fast food.

---

### 3.3 Watson Discovery

**Descrição.** Motor de busca cognitiva e análise de conteúdo que aplica enriquecimentos NLP (extração de entidades, sentimento, emoção, palavras-chave, classificação de categorias, etc.) sobre coleções de documentos estruturados e não estruturados.

**Endpoint (consulta a um projeto):**
```
POST {URL}/v2/projects/{project_id}/query?version=2023-03-31
```

**Exemplo (cURL):**
```bash
curl -X POST \
  -u "apikey:$APIKEY" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "política de devolução",
    "count": 5
  }' \
  "$URL/v2/projects/$PROJECT_ID/query?version=2023-03-31"
```

**Aplicação prática.** Busca semântica em uma base de contratos jurídicos para localizar cláusulas relacionadas a um tema, mesmo quando o vocabulário exato difere entre documentos.

---

### 3.4 Watson Speech to Text

**Descrição.** API de reconhecimento de fala que transcreve áudio para texto em diversos idiomas e formatos de áudio, retornando o conteúdo em UTF-8.

**Endpoint:**
```
POST {URL}/v1/recognize
```

**Exemplo (cURL):**
```bash
curl -X POST \
  -u "apikey:$APIKEY" \
  -H "Content-Type: audio/wav" \
  --data-binary @audio.wav \
  "$URL/v1/recognize?model=pt-BR_Multimedia"
```

**Aplicação prática.** Transcrição automática de ligações de call center para análise de qualidade e mineração de tópicos recorrentes.

---

### 3.5 Watson Text to Speech

**Descrição.** Converte texto escrito em áudio com vozes naturais em diferentes idiomas e estilos, retornando arquivos de áudio (ex.: `audio/wav`, `audio/ogg`).

**Endpoint:**
```
POST {URL}/v1/synthesize
```

**Exemplo (cURL):**
```bash
curl -X POST \
  -u "apikey:$APIKEY" \
  -H "Content-Type: application/json" \
  -H "Accept: audio/wav" \
  -d '{ "text": "Olá, seja bem-vindo ao nosso serviço." }' \
  "$URL/v1/synthesize?voice=pt-BR_IsabelaV3Voice" \
  --output saida.wav
```

**Aplicação prática.** Geração de áudios para um aplicativo de acessibilidade que lê notícias em voz alta para usuários com deficiência visual.

---

### 3.6 Watson Language Translator

**Descrição.** Serviço de tradução automática entre pares de idiomas via modelos pré-treinados (`model_id`, ex.: `en-pt`, `en-es`).

**Endpoint:**
```
POST {URL}/v3/translate?version=2018-05-01
```

**Exemplo (cURL):**
```bash
curl -X POST \
  -u "apikey:$APIKEY" \
  -H "Content-Type: application/json" \
  -d '{
    "text": ["Hello, world!", "How are you?"],
    "model_id": "en-pt"
  }' \
  "$URL/v3/translate?version=2018-05-01"
```

**Resposta esperada (resumida):**
```json
{
  "translations": [
    { "translation": "Olá, mundo!" },
    { "translation": "Como você está?" }
  ],
  "word_count": 5,
  "character_count": 26
}
```

**Aplicação prática.** Tradução automática de descrições de produtos em catálogo de e-commerce para múltiplos mercados.

> ⚠️ Verifique a disponibilidade atual do serviço no catálogo da IBM Cloud antes de usar em produção.

---

### 3.7 watsonx.ai (Foundation Models)

**Descrição.** Plataforma para inferência sobre modelos de fundação (LLMs), incluindo geração de texto, sumarização, classificação e *prompt tuning*. Diferente dos serviços Watson clássicos, exige autenticação via **Bearer token IAM**.

**Obtenção do token IAM:**
```bash
TOKEN=$(curl -X POST \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "grant_type=urn:ibm:params:oauth:grant-type:apikey&apikey=$APIKEY" \
  "https://iam.cloud.ibm.com/identity/token" | jq -r .access_token)
```

**Endpoint (geração de texto):**
```
POST https://{region}.ml.cloud.ibm.com/ml/v1/text/generation?version=2023-05-29
```

**Exemplo (cURL):**
```bash
curl -X POST \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "ibm/granite-13b-instruct-v2",
    "input": "Resuma em uma frase: O Brasil é o maior país da América do Sul...",
    "parameters": { "max_new_tokens": 60 },
    "project_id": "'"$PROJECT_ID"'"
  }' \
  "https://us-south.ml.cloud.ibm.com/ml/v1/text/generation?version=2023-05-29"
```

**Aplicação prática.** Sumarização automática de relatórios extensos em parágrafos curtos para envio em e-mails executivos.

---

## 4. Conclusão

O levantamento dos serviços de NLP da IBM Cloud revela que problemas de monitoramento de sentimento em escala raramente são resolvidos por um único modelo monolítico, mas sim por uma arquitetura modular de componentes especializados — extração de entidades e palavras-chave (NLU), classificação de sentimento e emoção, busca semântica sobre grandes volumes (Discovery), transcrição de áudio (Speech to Text) e LLMs para sumarização e geração de insights (watsonx.ai). Aplicado ao nosso projeto, podemos ver que esse insight valida um caminho claro que já estamos seguindo: módulos adicionais — reconhecimento da marca/produto mencionado, extração de tópicos recorrentes, agrupamento semântico de reclamações similares e sumarização automática de picos de menções — são plugados incrementalmente sem reescrever o core do nosso sistema. Essa separação em microserviços de NLP, combinada com ingestão multicanal padronizada (Google Avaliações, X, Instagram, Reclame Aqui) e armazenamento estruturado dos resultados, é o que viabiliza escalar milhares de registros em tempo quase real e contínuo das quatro marcas (Burger King, Popeyes, Subway e Starbucks) mantendo custo computacional, manutenibilidade e capacidade de adicionar novas análises sob demanda.

---

## 5. Referências

Todas as informações deste levantamento foram obtidas das fontes oficiais da IBM e da documentação pública dos SDKs. Fontes consultadas:

1. **IBM Cloud — Catálogo de serviços (categoria AI).** Disponível em: https://cloud.ibm.com/catalog?category=ai
2. **IBM Cloud — Natural Language Understanding (catálogo).** https://cloud.ibm.com/catalog/services/natural-language-understanding
3. **IBM Cloud API Docs — Natural Language Understanding.** https://cloud.ibm.com/apidocs/natural-language-understanding
4. **IBM Cloud — Getting started with Natural Language Understanding.** https://cloud.ibm.com/docs/natural-language-understanding?topic=natural-language-understanding-getting-started
5. **IBM — Página de produto: Natural Language Understanding.** https://www.ibm.com/products/natural-language-understanding
6. **IBM Cloud API Docs — watsonx Assistant v2.** https://cloud.ibm.com/apidocs/assistant-v2
7. **IBM Cloud Docs — watsonx Assistant API overview.** https://cloud.ibm.com/docs/watson-assistant?topic=watson-assistant-api-overview
8. **IBM Cloud — watsonx Assistant (catálogo).** https://cloud.ibm.com/catalog/services/watsonx-assistant
9. **IBM Cloud API Docs — Discovery v2.** https://cloud.ibm.com/apidocs/discovery-data
10. **IBM — Página de produto: Watson Discovery.** https://www.ibm.com/products/watson-discovery
11. **IBM Cloud API Docs — Speech to Text.** https://cloud.ibm.com/apidocs/speech-to-text/speech-to-text-icp
12. **IBM — Página de produto: Text to Speech.** https://www.ibm.com/in-en/cloud/watson-text-to-speech
13. **IBM/watson-translator-101 — Exemplo oficial de uso da API Translator.** https://github.com/IBM/watson-translator-101/blob/master/translation.md
14. **IBM Cloud API Docs — watsonx.ai.** https://cloud.ibm.com/apidocs/watsonx-ai
15. **IBM — Página de produto: watsonx.** https://www.ibm.com/products/watsonx
16. **IBM Developer — Watson APIs.** https://developer.ibm.com/components/watson-apis/
17. **IBM Watson APIs — Organização oficial no GitHub (SDKs).** https://github.com/watson-developer-cloud
18. **IBM — Soluções de NLP.** https://www.ibm.com/solutions/natural-language-processing

> *Acesso às fontes: junho de 2026.*


**FOI UTILIZADA IA GENERATIVA PARA: 
- PLANEJAMENTO DE PESQUISA;
- ORTOGRAFIA E SEMÂNTICA;
- ESTUDO DAS OPÇÕES DE CLOUD;
- BUSCA DAS DOCUMENTAÇÕES NECESSÁRIAS;
- COMPREENSÃO DAS REQUISIÇÕES DAS FERRAMENTAS.